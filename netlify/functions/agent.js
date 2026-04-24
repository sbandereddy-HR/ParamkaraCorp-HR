// ParamkaraCorp-HR Agent — Netlify Serverless Function
// Groq-powered, auto-detects document type and routes to correct analysis

const GROQ_API_KEY = process.env.GROQ_API_KEY;
const MODEL = "llama-3.3-70b-versatile";

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

// ─── Groq caller ──────────────────────────────────────────────────────────────
async function groq(messages, json = false, maxTokens = 4096) {
  const body = {
    model: MODEL,
    messages,
    temperature: 0.1,
    max_tokens: maxTokens,
  };
  if (json) body.response_format = { type: "json_object" };

  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${GROQ_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Groq error ${res.status}: ${err}`);
  }

  const data = await res.json();
  const content = data.choices?.[0]?.message?.content ?? "";
  if (json) {
    const match = content.match(/\{[\s\S]*\}/);
    return match ? JSON.parse(match[0]) : JSON.parse(content);
  }
  return content;
}

// ─── Document type detector ───────────────────────────────────────────────────
async function detectDocumentType(text, fileName) {
  const lower = text.toLowerCase();
  const fileL = (fileName || "").toLowerCase();

  // Fast heuristic first
  if (
    fileL.includes("jd") ||
    fileL.includes("job") ||
    lower.includes("job description") ||
    lower.includes("we are looking for") ||
    lower.includes("responsibilities") && lower.includes("requirements") && lower.includes("qualifications")
  ) return "jd_resume";

  if (
    lower.includes("offer letter") ||
    lower.includes("we are pleased to offer") ||
    lower.includes("joining date") ||
    lower.includes("cost to company") && lower.includes("designation")
  ) return "offer_letter";

  if (
    lower.includes("salary slip") ||
    lower.includes("payslip") ||
    lower.includes("pay slip") ||
    lower.includes("gross earnings") ||
    lower.includes("net pay") ||
    (lower.includes("basic") && lower.includes("hra") && lower.includes("pf"))
  ) return "salary_slip";

  if (
    lower.includes("epfo") ||
    lower.includes("service history") ||
    lower.includes("experience letter") ||
    lower.includes("employment certificate") ||
    lower.includes("this is to certify") && lower.includes("employed")
  ) return "employment";

  // AI fallback for ambiguous docs
  const result = await groq([{
    role: "user",
    content: `Classify this document. Return ONLY valid JSON: {"type": "offer_letter"|"salary_slip"|"jd_resume"|"employment"|"unknown", "reason": "one line"}

Document (first 600 chars): ${text.slice(0, 600)}`,
  }], true, 100);

  return result.type || "unknown";
}

// ─── GST checksum validator ───────────────────────────────────────────────────
function validateGST(text) {
  const m = text.match(/\b([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1})\b/);
  if (!m) return { found: false, verdict: "NOT_FOUND" };
  const gst = m[1];
  const CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  let factor = 1, sum = 0;
  for (let i = 0; i < 14; i++) {
    let code = CHARS.indexOf(gst[i]);
    let digit = factor * code;
    digit = Math.floor(digit / 36) + (digit % 36);
    sum += digit;
    factor = factor === 1 ? 3 : 1;
  }
  const expected = CHARS[(36 - (sum % 36)) % 36];
  return {
    found: true,
    raw: gst,
    valid: gst[14] === expected,
    verdict: gst[14] === expected ? "VALID" : "INVALID_CHECKSUM",
  };
}

// ─── CIN validator ────────────────────────────────────────────────────────────
function validateCIN(text) {
  const m = text.match(/\b([LUlu][0-9]{5}[A-Za-z]{2}[0-9]{4}[A-Za-z]{3}[0-9]{6})\b/);
  if (!m) return { found: false, verdict: "NOT_FOUND" };
  const cin = m[1].toUpperCase();
  const stateCode = cin.slice(6, 8);
  const year = parseInt(cin.slice(8, 12));
  const companyType = cin.slice(12, 15);
  const VALID_STATES = new Set(["AN","AP","AR","AS","BR","CH","CG","DD","DL","DN","GA","GJ","HP","HR","JH","JK","KA","KL","LA","LD","MH","ML","MN","MP","MZ","NL","OD","PB","PY","RJ","SK","TN","TR","TS","UK","UP","WB"]);
  const VALID_TYPES = new Set(["PLC","PTC","GOI","SGC","FLC","FTC","NPL","ULL","ULT","GAP","GAT"]);
  const valid = VALID_STATES.has(stateCode) && year >= 1850 && year <= new Date().getFullYear() && VALID_TYPES.has(companyType);
  return { found: true, raw: cin, valid, stateCode, year, companyType, verdict: valid ? "VALID" : "INVALID_FORMAT" };
}

// ─── Net pay math checker ─────────────────────────────────────────────────────
function extractAmt(text, ...labels) {
  for (const label of labels) {
    const esc = label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s*");
    const p = new RegExp(`${esc}\\s*[:\\-]?\\s*(?:INR|₹|Rs\\.?)?\\s*([0-9][0-9,]*)`, "i");
    const m = text.match(p);
    if (m?.[1]) {
      const v = parseFloat(m[1].replace(/,/g, ""));
      if (!isNaN(v) && v > 0) return v;
    }
  }
  return null;
}

// ─── Workflow: JD vs Resume ───────────────────────────────────────────────────
async function analyzeJDResume(documents) {
  const jdDoc = documents.find(d => d.detectedType === "jd_resume" && d.fileName.toLowerCase().includes("jd")) || documents[0];
  const resumeDoc = documents.find(d => d !== jdDoc) || documents[0];

  const result = await groq([{
    role: "user",
    content: `You are an expert HR analyst. Analyze the Job Description and Resume.
Return ONLY valid JSON:
{
  "candidate": {"name": "string", "total_experience_years": number},
  "jd_skills": [{"skill": "string", "required_years": number|null, "mandatory": boolean}],
  "skill_match": [{"skill": "string", "required_years": number|null, "actual_years": number, "matched": boolean, "gap": "string|null"}],
  "scorecard": {
    "match_percentage": number,
    "mandatory_skills_met": number,
    "mandatory_skills_total": number,
    "recommendation": "SHORTLIST"|"MAYBE"|"REJECT",
    "recommendation_reason": "string",
    "strengths": ["string"],
    "gaps": ["string"]
  }
}

JOB DESCRIPTION: ${(jdDoc?.text || documents[0]?.text || "").slice(0, 3000)}
RESUME: ${(resumeDoc?.text || documents[1]?.text || "").slice(0, 3000)}`,
  }], true);

  return { type: "jd_resume", ...result };
}

// ─── Workflow: Offer Letter ───────────────────────────────────────────────────
async function analyzeOfferLetter(text) {
  const cin = validateCIN(text);
  const gst = validateGST(text);

  const isMNC = /accenture|opentext|cognizant|capgemini|oracle|infosys|wipro|tcs\b|ibm\b|deloitte/i.test(text);
  const letterYear = (() => {
    const m = text.match(/\b(20[012]\d|19[89]\d)\b/);
    return m ? parseInt(m[1]) : null;
  })();

  const result = await groq([{
    role: "user",
    content: `You are a STRICT fraud detection expert for Indian offer letters.

CIN CHECK: ${cin.found ? `${cin.raw} — ${cin.verdict}` : "NOT FOUND"}
GST CHECK: ${gst.found ? `${gst.raw} — ${gst.verdict}` : "NOT FOUND"}
LETTER YEAR: ${letterYear || "unknown"}
IS MNC: ${isMNC}

Rules:
- Pre-2017 letters: GST absence is NORMAL (GST didn't exist)
- MNC letters: CIN absence is NORMAL
- INVALID_CHECKSUM GST = FAKE
- INVALID_FORMAT CIN = FAKE

Return ONLY valid JSON:
{
  "overall_score": number,
  "verdict": "AUTHENTIC"|"SUSPICIOUS"|"FAKE",
  "verdict_reason": "string",
  "checks": {
    "company_legitimacy": {"score": number, "status": "PASS"|"WARN"|"FAIL", "finding": "string"},
    "ctc_format": {"score": number, "status": "PASS"|"WARN"|"FAIL", "finding": "string"},
    "designation_validity": {"score": number, "status": "PASS"|"WARN"|"FAIL", "finding": "string"},
    "dates_logic": {"score": number, "status": "PASS"|"WARN"|"FAIL", "finding": "string"},
    "language_quality": {"score": number, "status": "PASS"|"WARN"|"FAIL", "finding": "string"},
    "red_flags": {"score": number, "status": "PASS"|"WARN"|"FAIL", "finding": "string"},
    "ai_content_risk": {"score": number, "status": "PASS"|"WARN"|"FAIL", "finding": "string"}
  },
  "red_flags_list": ["string"],
  "positive_signals": ["string"]
}

OFFER LETTER:
${text.slice(0, 5000)}`,
  }], true);

  return { type: "offer_letter", cin_validation: cin, gst_validation: gst, ...result };
}

// ─── Workflow: Salary Slip ────────────────────────────────────────────────────
async function analyzeSalarySlip(text, claimedCTC) {
  const gst = validateGST(text);
  const gross = extractAmt(text, "Gross Earnings", "Total Earnings", "Gross Salary", "Gross");
  const basic = extractAmt(text, "Basic Salary", "Basic Pay", "Basic");
  const pf = extractAmt(text, "Provident Fund", "PF", "EPF", "Employee PF", "PF Deduction");
  const deductions = extractAmt(text, "Total Deductions", "Gross Deductions", "Net Deductions");
  const net = extractAmt(text, "Net Pay", "Net Salary", "Take Home", "Net Amount Payable");

  // Math checks
  const pfExpected = basic ? basic * 0.12 : null;
  const pfOk = pf && pfExpected ? (pf === 1800 || Math.abs(pf - pfExpected) <= Math.max(50, pfExpected * 0.01)) : null;
  const netOk = gross && deductions && net ? Math.abs(net - (gross - deductions)) <= Math.max(10, gross * 0.002) : null;

  // CTC inflation
  let ctcInflation = null;
  if (gross) {
    const employerPF = basic ? Math.min(basic * 0.12, 1800) : 1800;
    const gratuity = basic ? basic * 0.0481 : 0;
    const derivedAnnual = Math.round((gross + employerPF + gratuity) * 12);
    if (claimedCTC) {
      const claimed = claimedCTC < 500 ? claimedCTC * 100000 : claimedCTC;
      const gap = claimed - derivedAnnual;
      const pct = (gap / derivedAnnual) * 100;
      ctcInflation = {
        derived_annual_ctc: derivedAnnual,
        claimed_ctc_annual: claimed,
        inflation_gap: Math.round(gap),
        inflation_percent: Math.round(pct * 10) / 10,
        verdict: Math.abs(pct) <= 15 ? "NORMAL" : pct > 30 ? "HIGHLY_INFLATED" : "INFLATED",
      };
    } else {
      ctcInflation = { derived_annual_ctc: derivedAnnual, verdict: "NO_CLAIM_TO_COMPARE" };
    }
  }

  const result = await groq([{
    role: "user",
    content: `You are an Indian payroll fraud investigator.

PRE-COMPUTED CHECKS:
- GST: ${gst.found ? `${gst.raw} — ${gst.verdict}` : "NOT FOUND"}
- Gross: ₹${gross || "not found"}
- Basic: ₹${basic || "not found"}
- PF: ₹${pf || "not found"} (expected 12% = ₹${pfExpected?.toFixed(0) || "?"}) → ${pfOk === null ? "insufficient data" : pfOk ? "VALID" : "MISMATCH"}
- Net pay math: ${netOk === null ? "insufficient data" : netOk ? "VALID" : "MISMATCH — POSSIBLE TAMPERING"}

Return ONLY valid JSON:
{
  "overall_score": number,
  "verdict": "GENUINE"|"SUSPICIOUS"|"FAKE",
  "verdict_reason": "string",
  "rule_checks": [{"name": "string", "status": "pass"|"fail"|"neutral", "finding": "string"}],
  "ai_reasoning": "string",
  "ai_generation_risk": number,
  "red_flags": ["string"],
  "positive_signals": ["string"]
}

SALARY SLIP:
${text.slice(0, 4000)}`,
  }], true);

  return { type: "salary_slip", ctc_inflation: ctcInflation, math_checks: { pf_valid: pfOk, net_valid: netOk, gross, basic, pf, net }, gst_validation: gst, ...result };
}

// ─── Workflow: Employment History ─────────────────────────────────────────────
async function analyzeEmployment(documents, mode) {
  const resume = documents.find(d => d.label === "resume") || documents[0];
  const other = documents.find(d => d !== resume) || documents[1];

  if (mode === "gap_analysis" || documents.length === 1) {
    const result = await groq([{
      role: "user",
      content: `Extract all jobs and calculate gaps. Return ONLY valid JSON:
{
  "companies": [{"company": "string", "from": "YYYY-MM", "to": "YYYY-MM|Present", "role": "string"}],
  "gaps": [{"from_company": "string", "from_date": "YYYY-MM", "to_company": "string", "to_date": "YYYY-MM", "gap_months": number, "severity": "SHORT"|"MEDIUM"|"LONG", "label": "string"}],
  "total_gap_months": number,
  "verdict": "CLEAN"|"GAPS_FOUND",
  "verdict_reason": "string"
}
RESUME: ${resume?.text?.slice(0, 3000) || ""}`,
    }], true);
    return { type: "employment", mode: "gap_analysis", ...result };
  }

  const isEPFO = other?.text?.toLowerCase().includes("epfo") || other?.text?.toLowerCase().includes("service history");

  const result = await groq([{
    role: "user",
    content: `${isEPFO ? "Cross-verify resume against EPFO service history" : "Compare resume with experience letter"}.
Return ONLY valid JSON:
{
  "matched": [{"resume_company": "string", "verified_company": "string", "date_match": boolean, "gap_months": number}],
  "only_in_resume": [{"company": "string", "flag": "string"}],
  "only_in_verified": [{"company": "string"}],
  "experience_summary": {"resume_total_years": number, "verified_total_years": number, "inflation_years": number},
  "verdict": "MATCH"|"PARTIAL"|"MISMATCH",
  "verdict_reason": "string",
  "red_flags": ["string"]
}
RESUME: ${resume?.text?.slice(0, 2500) || ""}
${isEPFO ? "EPFO SERVICE HISTORY" : "EXPERIENCE LETTER"}: ${other?.text?.slice(0, 2500) || ""}`,
  }], true);
  return { type: "employment", mode: isEPFO ? "epfo" : "experience_letter", ...result };
}

// ─── Conversational fallback ──────────────────────────────────────────────────
async function conversationalReply(userMessage, history) {
  const messages = [
    {
      role: "system",
      content: `You are ParamkaraCorp-HR Assistant, an expert AI agent for Indian HR and recruitment verification. You help HR professionals verify:
1. JD vs Resume matching (skill gap, recommendation)
2. Offer letter fraud detection (CIN, GST, email domain)
3. Salary slip authenticity (PF math, CTC inflation)
4. Employment history (EPFO cross-check, gap analysis)

Be concise, professional, and helpful. When users upload documents, tell them you'll analyze them. Format responses in clean markdown.`,
    },
    ...history.slice(-6),
    { role: "user", content: userMessage },
  ];
  return await groq(messages, false, 512);
}

// ─── Main handler ─────────────────────────────────────────────────────────────
exports.handler = async (event) => {
  if (event.httpMethod === "OPTIONS") {
    return { statusCode: 200, headers: CORS, body: "" };
  }

  if (event.httpMethod !== "POST") {
    return { statusCode: 405, headers: CORS, body: JSON.stringify({ error: "Method not allowed" }) };
  }

  if (!GROQ_API_KEY) {
    return { statusCode: 500, headers: { ...CORS, "Content-Type": "application/json" }, body: JSON.stringify({ error: "GROQ_API_KEY not configured in Netlify environment variables" }) };
  }

  try {
    const body = JSON.parse(event.body || "{}");
    const { message, documents, history, claimedCTC, mode } = body;

    // No documents → conversational
    if (!documents || documents.length === 0) {
      const reply = await conversationalReply(message || "Hello", history || []);
      return {
        statusCode: 200,
        headers: { ...CORS, "Content-Type": "application/json" },
        body: JSON.stringify({ type: "conversation", reply }),
      };
    }

    // Detect types for all uploaded docs
    for (const doc of documents) {
      if (!doc.detectedType) {
        doc.detectedType = await detectDocumentType(doc.text, doc.fileName);
      }
    }

    const types = documents.map(d => d.detectedType);
    const hasOfferLetter = types.includes("offer_letter");
    const hasSalarySlip = types.includes("salary_slip");
    const hasEmployment = types.includes("employment") || types.includes("jd_resume") && documents.length > 1;
    const isJDResume = types.filter(t => t === "jd_resume").length >= 2 || (message || "").toLowerCase().includes("match");

    let result;
    if (isJDResume) {
      result = await analyzeJDResume(documents);
    } else if (hasOfferLetter) {
      const doc = documents.find(d => d.detectedType === "offer_letter");
      result = await analyzeOfferLetter(doc.text);
    } else if (hasSalarySlip) {
      const doc = documents.find(d => d.detectedType === "salary_slip");
      result = await analyzeSalarySlip(doc.text, claimedCTC);
    } else if (hasEmployment || documents.length >= 2) {
      result = await analyzeEmployment(documents, mode);
    } else {
      // Single unknown doc — let AI figure it out
      const detectedType = documents[0].detectedType;
      if (detectedType === "offer_letter") result = await analyzeOfferLetter(documents[0].text);
      else if (detectedType === "salary_slip") result = await analyzeSalarySlip(documents[0].text, claimedCTC);
      else {
        const reply = await conversationalReply(
          `The user uploaded a document named "${documents[0].fileName}". Text preview: ${documents[0].text.slice(0, 400)}. Tell them what you detected and what you can do.`,
          history || []
        );
        result = { type: "conversation", reply };
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
      body: JSON.stringify({ error: err.message || "Internal error" }),
    };
  }
};
