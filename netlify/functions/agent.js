// ParamkaraCorp-HR Agent — Netlify Serverless Function
// Single-call architecture: one request → one Groq call → structured result
// Supports: PDF text, DOCX text, images (vision), multi-doc cross-check

const GROQ_API_KEY = process.env.GROQ_API_KEY;
const MODEL        = "llama-3.3-70b-versatile";
// For image/vision payloads, use llama-4-scout or groq vision model if available
const VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct";

const CORS = {
  "Access-Control-Allow-Origin":  "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

// ── Groq caller ───────────────────────────────────────────────────────────────
async function groq(messages, json = false, maxTokens = 3000) {
  const body = {
    model: MODEL,
    messages,
    temperature: 0.05,
    max_tokens: maxTokens,
  };
  if (json) body.response_format = { type: "json_object" };

  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization:  `Bearer ${GROQ_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Groq error ${res.status}: ${err}`);
  }

  const data    = await res.json();
  const content = data.choices?.[0]?.message?.content ?? "";
  if (json) {
    const match = content.match(/\{[\s\S]*\}/);
    try { return match ? JSON.parse(match[0]) : JSON.parse(content); }
    catch { throw new Error("JSON parse failed: " + content.slice(0, 200)); }
  }
  return content;
}

// ── Validators (zero API calls) ───────────────────────────────────────────────
function validateGST(text) {
  const m = text.match(/\b([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1})\b/);
  if (!m) return { found: false, verdict: "NOT_FOUND" };
  const gst   = m[1];
  const CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  let factor  = 1, sum = 0;
  for (let i = 0; i < 14; i++) {
    let digit = factor * CHARS.indexOf(gst[i]);
    digit     = Math.floor(digit / 36) + (digit % 36);
    sum      += digit;
    factor    = factor === 1 ? 3 : 1;
  }
  const expected = CHARS[(36 - (sum % 36)) % 36];
  return {
    found: true, raw: gst,
    valid: gst[14] === expected,
    verdict: gst[14] === expected ? "VALID" : "INVALID_CHECKSUM",
  };
}

function validateCIN(text) {
  const m = text.match(/\b([LUlu][0-9]{5}[A-Za-z]{2}[0-9]{4}[A-Za-z]{3}[0-9]{6})\b/);
  if (!m) return { found: false, verdict: "NOT_FOUND" };
  const cin         = m[1].toUpperCase();
  const stateCode   = cin.slice(6, 8);
  const year        = parseInt(cin.slice(8, 12));
  const companyType = cin.slice(12, 15);
  const VALID_STATES = new Set(["AN","AP","AR","AS","BR","CH","CG","DD","DL","DN","GA","GJ","HP","HR","JH","JK","KA","KL","LA","LD","MH","ML","MN","MP","MZ","NL","OD","PB","PY","RJ","SK","TN","TR","TS","UK","UP","WB"]);
  const VALID_TYPES  = new Set(["PLC","PTC","GOI","SGC","FLC","FTC","NPL","ULL","ULT","GAP","GAT"]);
  const valid = VALID_STATES.has(stateCode) && year >= 1850 && year <= new Date().getFullYear() && VALID_TYPES.has(companyType);
  return { found: true, raw: cin, valid, stateCode, year, companyType, verdict: valid ? "VALID" : "INVALID_FORMAT" };
}

function extractAmt(text, ...labels) {
  for (const label of labels) {
    const esc = label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s*");
    const m   = text.match(new RegExp(`${esc}\\s*[:\\-]?\\s*(?:INR|₹|Rs\\.?)?\\s*([0-9][0-9,]*)`, "i"));
    if (m?.[1]) {
      const v = parseFloat(m[1].replace(/,/g, ""));
      if (!isNaN(v) && v > 0) return v;
    }
  }
  return null;
}

// ── Document type detector (heuristic, zero API calls) ───────────────────────
function detectDocType(text, fileName) {
  const lower = (text || "").toLowerCase();
  const fileL = (fileName || "").toLowerCase();

  if (fileL.includes("jd") || fileL.includes("job_desc") ||
      lower.includes("job description") || lower.includes("we are looking for") ||
      (lower.includes("responsibilities") && lower.includes("requirements")))
    return "jd_resume";

  if (lower.includes("offer letter") || lower.includes("we are pleased to offer") ||
      lower.includes("joining date") ||
      (lower.includes("cost to company") && lower.includes("designation")))
    return "offer_letter";

  if (lower.includes("salary slip") || lower.includes("payslip") || lower.includes("pay slip") ||
      lower.includes("gross earnings") || lower.includes("net pay") ||
      (lower.includes("basic") && lower.includes("hra") && lower.includes("pf")))
    return "salary_slip";

  if (lower.includes("epfo") || lower.includes("service history") ||
      lower.includes("experience letter") || lower.includes("employment certificate") ||
      (lower.includes("this is to certify") && lower.includes("employed")))
    return "employment";

  if (lower.includes("curriculum vitae") || lower.includes("objective") ||
      (lower.includes("experience") && lower.includes("education") && lower.includes("skills")))
    return "employment";

  return "unknown";
}

// ── Shared prompt builder — puts ALL docs in ONE prompt ──────────────────────
// This is the key to single-call architecture
function buildUnifiedPrompt(documents, claimedCTC, mode, userMessage) {
  const types    = documents.map(d => d.detectedType || detectDocType(d.text, d.fileName));
  const hasSalary = types.includes("salary_slip");
  const hasOffer  = types.includes("offer_letter");
  const hasJD     = types.filter(t => t === "jd_resume").length >= 1;
  const allJD     = types.filter(t => t === "jd_resume").length >= 2;
  const hasResume = types.some(t => t === "employment" || t === "jd_resume");
  const hasEPFO   = documents.some(d => (d.text||"").toLowerCase().includes("epfo") || (d.text||"").toLowerCase().includes("service history"));
  const isMoonlighting = mode === "moonlighting";

  // Pre-compute validators
  let validatorCtx = "";
  if (hasOffer) {
    const offerDoc = documents.find(d => d.detectedType === "offer_letter");
    if (offerDoc) {
      const cin = validateCIN(offerDoc.text);
      const gst = validateGST(offerDoc.text);
      const isMNC = /accenture|opentext|cognizant|capgemini|oracle|infosys|wipro|tcs\b|ibm\b|deloitte/i.test(offerDoc.text);
      const year  = (offerDoc.text.match(/\b(20[012]\d|19[89]\d)\b/) || [])[1];
      validatorCtx = `PRE-COMPUTED (do NOT re-validate):
CIN: ${cin.found ? `${cin.raw} → ${cin.verdict}` : "NOT FOUND"}
GST: ${gst.found ? `${gst.raw} → ${gst.verdict}` : "NOT FOUND"}
IS_MNC: ${isMNC} | YEAR: ${year || "unknown"}
NOTE: Pre-2017 letters → GST absence is NORMAL. MNC → CIN absence is NORMAL. INVALID_CHECKSUM = FAKE.\n\n`;
    }
  }

  if (hasSalary) {
    const salDoc = documents.find(d => d.detectedType === "salary_slip");
    if (salDoc) {
      const gst    = validateGST(salDoc.text);
      const gross  = extractAmt(salDoc.text, "Gross Earnings", "Total Earnings", "Gross Salary", "Gross");
      const basic  = extractAmt(salDoc.text, "Basic Salary", "Basic Pay", "Basic");
      const pf     = extractAmt(salDoc.text, "Provident Fund", "PF", "EPF", "Employee PF");
      const deductions = extractAmt(salDoc.text, "Total Deductions", "Net Deductions");
      const net    = extractAmt(salDoc.text, "Net Pay", "Net Salary", "Take Home");
      const pfExp  = basic ? basic * 0.12 : null;
      const pfOk   = pf && pfExp ? (pf === 1800 || Math.abs(pf - pfExp) <= Math.max(50, pfExp * 0.01)) : null;
      const netOk  = gross && deductions && net ? Math.abs(net - (gross - deductions)) <= Math.max(10, gross * 0.002) : null;
      validatorCtx += `SALARY PRE-COMPUTED:
GST: ${gst.found ? `${gst.raw} → ${gst.verdict}` : "NOT FOUND"}
Gross=₹${gross||"?"}, Basic=₹${basic||"?"}, PF=₹${pf||"?"} (expected 12%=₹${pfExp?.toFixed(0)||"?"}) → PF_VALID:${pfOk===null?"?":pfOk}
Net math: ${netOk===null?"?":netOk?"VALID":"MISMATCH — possible tampering"}\n`;
      if (claimedCTC && gross) {
        const employerPF   = basic ? Math.min(basic * 0.12, 1800) : 1800;
        const gratuity     = basic ? basic * 0.0481 : 0;
        const derivedAnn   = Math.round((gross + employerPF + gratuity) * 12);
        const claimed      = claimedCTC < 500 ? claimedCTC * 100000 : claimedCTC;
        const gap          = claimed - derivedAnn;
        const inflatePct   = Math.round((gap / derivedAnn) * 1000) / 10;
        validatorCtx += `CTC: Derived=₹${(derivedAnn/100000).toFixed(2)}L, Claimed=₹${(claimed/100000).toFixed(2)}L, Gap=${inflatePct>0?'+':''}${inflatePct}% → ${Math.abs(inflatePct)<=15?"NORMAL":inflatePct>30?"HIGHLY_INFLATED":"INFLATED"}\n`;
      }
    }
  }

  // Determine analysis type
  let analysisType, jsonSchema;

  if (isMoonlighting) {
    analysisType = "moonlighting";
    jsonSchema = `{
  "type": "moonlighting",
  "verdict": "LOW_RISK"|"MEDIUM_RISK"|"HIGH_RISK",
  "verdict_reason": "string",
  "signals": [{"signal": "string", "risk": "LOW"|"MEDIUM"|"HIGH"}],
  "recommendations": ["string"]
}`;
  } else if (allJD || (hasJD && hasResume && documents.length >= 2)) {
    analysisType = "jd_resume";
    jsonSchema = `{
  "type": "jd_resume",
  "candidate": {"name": "string", "total_experience_years": number},
  "jd_skills": [{"skill": "string", "required_years": number|null, "mandatory": boolean}],
  "skill_match": [{"skill": "string", "required_years": number|null, "actual_years": number, "matched": boolean}],
  "scorecard": {
    "match_percentage": number,
    "mandatory_skills_met": number,
    "mandatory_skills_total": number,
    "recommendation": "SHORTLIST"|"MAYBE"|"REJECT",
    "recommendation_reason": "string",
    "strengths": ["string"],
    "gaps": ["string"]
  }
}`;
  } else if (hasOffer) {
    analysisType = "offer_letter";
    jsonSchema = `{
  "type": "offer_letter",
  "overall_score": number,
  "verdict": "AUTHENTIC"|"SUSPICIOUS"|"FAKE",
  "verdict_reason": "string",
  "checks": {
    "company_legitimacy": {"score": number, "status": "PASS"|"WARN"|"FAIL", "finding": "string"},
    "ctc_format": {"score": number, "status": "PASS"|"WARN"|"FAIL", "finding": "string"},
    "designation_validity": {"score": number, "status": "PASS"|"WARN"|"FAIL", "finding": "string"},
    "dates_logic": {"score": number, "status": "PASS"|"WARN"|"FAIL", "finding": "string"},
    "language_quality": {"score": number, "status": "PASS"|"WARN"|"FAIL", "finding": "string"},
    "red_flags": {"score": number, "status": "PASS"|"WARN"|"FAIL", "finding": "string"}
  },
  "red_flags_list": ["string"],
  "positive_signals": ["string"]
}`;
  } else if (hasSalary) {
    analysisType = "salary_slip";
    jsonSchema = `{
  "type": "salary_slip",
  "overall_score": number,
  "verdict": "GENUINE"|"SUSPICIOUS"|"FAKE",
  "verdict_reason": "string",
  "rule_checks": [{"name": "string", "status": "pass"|"fail"|"neutral", "finding": "string"}],
  "ai_reasoning": "string",
  "red_flags": ["string"],
  "positive_signals": ["string"],
  "ctc_inflation": {
    "derived_annual_ctc": number|null,
    "claimed_ctc_annual": number|null,
    "inflation_percent": number|null,
    "verdict": "NORMAL"|"INFLATED"|"HIGHLY_INFLATED"|"NO_CLAIM_TO_COMPARE"
  }
}`;
  } else if (hasResume && hasEPFO) {
    analysisType = "employment";
    jsonSchema = `{
  "type": "employment",
  "mode": "epfo_crosscheck",
  "matched": [{"resume_company": "string", "verified_company": "string", "date_match": boolean}],
  "only_in_resume": [{"company": "string", "flag": "string"}],
  "experience_summary": {"resume_total_years": number, "epfo_total_years": number, "inflation_years": number},
  "verdict": "MATCH"|"PARTIAL"|"MISMATCH",
  "verdict_reason": "string",
  "red_flags": ["string"]
}`;
  } else {
    // Single resume or unknown — gap analysis
    analysisType = "employment";
    jsonSchema = `{
  "type": "employment",
  "mode": "gap_analysis",
  "companies": [{"company": "string", "from": "YYYY-MM", "to": "YYYY-MM|Present", "role": "string"}],
  "gaps": [{"from_company": "string", "from_date": "YYYY-MM", "to_company": "string", "to_date": "YYYY-MM", "gap_months": number, "severity": "SHORT"|"MEDIUM"|"LONG", "label": "string"}],
  "total_gap_months": number,
  "verdict": "CLEAN"|"GAPS_FOUND",
  "verdict_reason": "string",
  "red_flags": ["string"]
}`;
  }

  // Build document sections (trim to save tokens)
  const docSections = documents.map((d, i) => {
    const label = (d.detectedType || detectDocType(d.text, d.fileName)).toUpperCase().replace("_", " ");
    if (d.base64Image) return `DOCUMENT ${i+1} (${d.fileName}): [IMAGE — analyze visually]`;
    return `--- DOCUMENT ${i+1}: ${d.fileName} [${label}] ---\n${(d.text || "").slice(0, 4000)}`;
  }).join("\n\n");

  const systemPrompt = `You are ParamkaraCorp-HR, an expert Indian HR fraud detection AI.
Analysis type: ${analysisType.toUpperCase()}
${validatorCtx}
RULES:
- Return ONLY valid JSON matching the schema below, no extra text
- Use pre-computed values as-is, do not re-derive them
- Be conservative: flag suspicious items as WARN not FAIL unless clearly fake
- For Indian context: know that PF ceiling is ₹1800/month (basic cap ₹15000), GST was introduced July 2017
- Score 0-100: 85+=PASS, 60-84=WARN, <60=FAIL

REQUIRED JSON SCHEMA:
${jsonSchema}`;

  const userPrompt = `${docSections}${userMessage ? `\n\nUser note: ${userMessage}` : ""}`;

  return { systemPrompt, userPrompt, analysisType };
}

// ── Conversational fallback ───────────────────────────────────────────────────
async function conversationalReply(userMessage, history) {
  const messages = [
    {
      role: "system",
      content: `You are ParamkaraCorp-HR Assistant. You help Indian HR professionals verify:
JD vs Resume matching, Offer letter fraud detection (CIN/GST), Salary slip authenticity (PF math, CTC inflation), Employment history (EPFO cross-check, gap analysis), and Moonlighting detection.
Be concise and professional. Format in markdown.`,
    },
    ...history.slice(-4),
    { role: "user", content: userMessage },
  ];
  const reply = await groq(messages, false, 400);
  return { type: "conversation", reply };
}

// ── Main handler ──────────────────────────────────────────────────────────────
exports.handler = async (event) => {
  if (event.httpMethod === "OPTIONS")
    return { statusCode: 200, headers: CORS, body: "" };

  if (event.httpMethod !== "POST")
    return { statusCode: 405, headers: CORS, body: JSON.stringify({ error: "Method not allowed" }) };

  if (!GROQ_API_KEY)
    return { statusCode: 500, headers: { ...CORS, "Content-Type": "application/json" },
             body: JSON.stringify({ error: "GROQ_API_KEY not configured in Netlify environment variables" }) };

  try {
    const body = JSON.parse(event.body || "{}");
    const { message, documents, history, claimedCTC, mode } = body;

    // ── No documents → conversational reply ──────────────────────────────────
    if (!documents || documents.length === 0) {
      const result = await conversationalReply(message || "Hello", history || []);
      return {
        statusCode: 200,
        headers: { ...CORS, "Content-Type": "application/json" },
        body: JSON.stringify(result),
      };
    }

    // ── Fill in detected types server-side if missing ─────────────────────────
    for (const doc of documents) {
      if (!doc.detectedType || doc.detectedType === "unknown") {
        doc.detectedType = detectDocType(doc.text || "", doc.fileName);
      }
    }

    // ── Build unified prompt (single call) ────────────────────────────────────
    const { systemPrompt, userPrompt, analysisType } = buildUnifiedPrompt(
      documents, claimedCTC, mode, message
    );

    // Check if any doc is an image (vision)
    const hasImage = documents.some(d => d.base64Image);

    let result;
    if (hasImage) {
      // Build vision-compatible message with image content blocks
      const contentBlocks = [];
      for (const doc of documents) {
        if (doc.base64Image) {
          contentBlocks.push({
            type: "image_url",
            image_url: { url: `data:${doc.imageMediaType};base64,${doc.base64Image}` },
          });
          contentBlocks.push({ type: "text", text: `Above image is: ${doc.fileName}` });
        } else if (doc.text) {
          contentBlocks.push({ type: "text", text: `--- ${doc.fileName} ---\n${doc.text.slice(0, 3000)}` });
        }
      }
      contentBlocks.push({ type: "text", text: `\nAnalyze these documents. Return JSON only:\n${systemPrompt.split("REQUIRED JSON SCHEMA:")[1] || ""}` });

      // Use vision model
      const visionBody = {
        model: VISION_MODEL,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: contentBlocks },
        ],
        temperature: 0.05,
        max_tokens: 2000,
        response_format: { type: "json_object" },
      };

      const visionRes = await fetch("https://api.groq.com/openai/v1/chat/completions", {
        method: "POST",
        headers: { Authorization: `Bearer ${GROQ_API_KEY}`, "Content-Type": "application/json" },
        body: JSON.stringify(visionBody),
      });

      if (!visionRes.ok) {
        // Fallback to text model with placeholder
        result = await groq([
          { role: "system", content: systemPrompt },
          { role: "user", content: "[Image uploaded — analyze based on file name and context]\n" + userPrompt },
        ], true, 2000);
      } else {
        const vd = await visionRes.json();
        const ct = vd.choices?.[0]?.message?.content ?? "{}";
        const match = ct.match(/\{[\s\S]*\}/);
        result = match ? JSON.parse(match[0]) : JSON.parse(ct);
      }
    } else {
      // ── Standard text-only single call ───────────────────────────────────
      result = await groq([
        { role: "system", content: systemPrompt },
        { role: "user",   content: userPrompt },
      ], true, 2500);
    }

    // Ensure type is set
    if (!result.type) result.type = analysisType;

    // Attach pre-computed CTC inflation for salary if present
    if (result.type === "salary_slip" && claimedCTC) {
      const salDoc = documents.find(d => d.detectedType === "salary_slip");
      if (salDoc) {
        const gross  = extractAmt(salDoc.text, "Gross Earnings", "Total Earnings", "Gross Salary", "Gross");
        const basic  = extractAmt(salDoc.text, "Basic Salary", "Basic Pay", "Basic");
        if (gross) {
          const employerPF  = basic ? Math.min(basic * 0.12, 1800) : 1800;
          const gratuity    = basic ? basic * 0.0481 : 0;
          const derivedAnn  = Math.round((gross + employerPF + gratuity) * 12);
          const claimed     = claimedCTC < 500 ? claimedCTC * 100000 : claimedCTC;
          const gap         = claimed - derivedAnn;
          const inflatePct  = Math.round((gap / derivedAnn) * 1000) / 10;
          result.ctc_inflation = {
            derived_annual_ctc: derivedAnn,
            claimed_ctc_annual: claimed,
            inflation_gap:     Math.round(gap),
            inflation_percent: inflatePct,
            verdict:           Math.abs(inflatePct) <= 15 ? "NORMAL" : inflatePct > 30 ? "HIGHLY_INFLATED" : "INFLATED",
          };
        }
      }
    }

    return {
      statusCode: 200,
      headers: { ...CORS, "Content-Type": "application/json" },
      body: JSON.stringify(result),
    };

  } catch (err) {
    console.error("Agent error:", err);
    return {
      statusCode: 500,
      headers: { ...CORS, "Content-Type": "application/json" },
      body: JSON.stringify({ error: err.message || "Internal server error" }),
    };
  }
};
