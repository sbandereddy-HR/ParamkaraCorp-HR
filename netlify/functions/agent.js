// ParamkaraCorp-HR Agent — Netlify Serverless Function
// Single-call architecture: one request → one Groq call → structured result
// Supports: PDF text, DOCX text, images (vision), multi-doc cross-check
 
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const MODEL        = "llama-3.3-70b-versatile";
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
  if (!text) return { found: false, verdict: "NOT_FOUND" };
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
  if (!text) return { found: false, verdict: "NOT_FOUND" };
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
  if (!text) return null;
  for (const label of labels) {
    const esc = label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s*");
    const m   = text.match(new RegExp(`${esc}\\s*[:\\-]?\\s*(?:INR|\u20b9|Rs\\.?)?\\s*([0-9][0-9,]*)`, "i"));
    if (m?.[1]) {
      const v = parseFloat(m[1].replace(/,/g, ""));
      if (!isNaN(v) && v > 0) return v;
    }
  }
  return null;
}
 
// ── FIX 5: Auto-extract CTC from document text ───────────────────────────────
function extractCTCFromText(text) {
  if (!text) return null;
  const patterns = [
    /(?:expected\s+ctc|ctc\s+offered|ctc|cost\s+to\s+company)\s*[:\-]?\s*(?:INR|\u20b9|Rs\.?)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:lpa|l\.p\.a|lakhs?(?:\s*per\s*annum)?|lac)/i,
    /(?:expected\s+ctc|ctc\s+offered|ctc|cost\s+to\s+company)\s*[:\-]?\s*(?:INR|\u20b9|Rs\.?)?\s*([0-9][0-9,]+)/i,
  ];
  for (const pat of patterns) {
    const m = text.match(pat);
    if (m?.[1]) {
      const raw = parseFloat(m[1].replace(/,/g, ""));
      if (!isNaN(raw) && raw > 0) {
        return raw < 500 ? raw * 100000 : raw;
      }
    }
  }
  return null;
}
 
// ── FIX 1: Document type detector — never returns undefined ──────────────────
function detectDocType(text, fileName) {
  try {
    const lower = (text || "").toLowerCase();
    const fileL = (fileName || "").toLowerCase();
 
    if (fileL.includes("jd") || fileL.includes("job_desc") || fileL.includes("job-desc") ||
        lower.includes("job description") || lower.includes("we are looking for") ||
        lower.includes("job title") || lower.includes("about the role") ||
        (lower.includes("responsibilities") && lower.includes("requirements")))
      return "jd";
 
    if (lower.includes("offer letter") || lower.includes("we are pleased to offer") ||
        lower.includes("joining date") ||
        (lower.includes("cost to company") && lower.includes("designation")))
      return "offer_letter";
 
    if (lower.includes("salary slip") || lower.includes("payslip") || lower.includes("pay slip") ||
        lower.includes("gross earnings") || lower.includes("net pay") ||
        (lower.includes("basic") && lower.includes("hra") && lower.includes("pf")))
      return "salary_slip";
 
    if (lower.includes("epfo") || lower.includes("service history") ||
        lower.includes("passbook") || lower.includes("uan") ||
        lower.includes("employees provident fund"))
      return "epfo";
 
    if (lower.includes("experience letter") || lower.includes("employment certificate") ||
        (lower.includes("this is to certify") && lower.includes("employed")))
      return "exp_letter";
 
    if (fileL.includes("resume") || fileL.includes("cv") ||
        lower.includes("curriculum vitae") ||
        (lower.includes("experience") && lower.includes("education") && lower.includes("skills")))
      return "resume";
 
    return "unknown";
  } catch (e) {
    console.error("detectDocType error:", e);
    return "unknown";
  }
}
 
// ── FIX 2 & 3: Explicit routing — no fall-through to conversation ─────────────
function determineAnalysisMode(documents) {
  const types   = documents.map(d => d.detectedType || "unknown");
  const has     = (t) => types.includes(t);
 
  const hasJD     = has("jd");
  const hasResume = has("resume") || has("exp_letter");
  const hasEPFO   = has("epfo");
  const hasSalary = has("salary_slip");
  const hasOffer  = has("offer_letter");
 
  // Moonlighting handled by caller (mode param)
  // Combined: JD + Resume + (EPFO or Salary) → run both analyses
  if (hasJD && hasResume && (hasEPFO || hasSalary)) return "combined_jd_employment";
  // JD + Resume
  if (hasJD && hasResume) return "jd_resume";
  // Single JD only — parse it, don't fall to chat
  if (hasJD) return "jd_only";
  // Offer + Salary together
  if (hasOffer && hasSalary) return "offer_salary";
  // Offer alone
  if (hasOffer) return "offer_letter";
  // Salary + Resume
  if (hasSalary && hasResume) return "salary_resume";
  // Salary alone
  if (hasSalary) return "salary_slip";
  // Resume + EPFO
  if (hasResume && hasEPFO) return "epfo_crosscheck";
  // EPFO alone
  if (hasEPFO) return "epfo_only";
  // Resume alone — gap analysis
  if (hasResume) return "resume_only";
  // Fallback: analyse whatever was sent
  return "unknown_docs";
}
 
// ── JSON Schemas ──────────────────────────────────────────────────────────────
const SCHEMAS = {
  moonlighting: `{
  "type": "moonlighting",
  "verdict": "LOW_RISK"|"MEDIUM_RISK"|"HIGH_RISK",
  "verdict_reason": "string",
  "signals": [{"signal": "string", "risk": "LOW"|"MEDIUM"|"HIGH"}],
  "recommendations": ["string"]
}`,
 
  jd_resume: `{
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
}`,
 
  jd_only: `{
  "type": "jd_only",
  "jd_summary": {"role": "string", "company": "string|null", "ctc_range": "string|null"},
  "required_skills": [{"skill": "string", "mandatory": boolean, "years": number|null}],
  "notice": "Resume not uploaded — upload candidate resume to run skill match scorecard"
}`,
 
  // FIX 3: Combined mode — two sections in one response
  combined_jd_employment: `{
  "type": "combined",
  "jd_match": {
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
  },
  "employment_check": {
    "mode": "epfo_crosscheck"|"gap_analysis",
    "companies": [{"company": "string", "from": "YYYY-MM", "to": "YYYY-MM|Present", "role": "string"}],
    "gaps": [{"from_company": "string", "from_date": "YYYY-MM", "to_company": "string", "to_date": "YYYY-MM", "gap_months": number, "severity": "SHORT"|"MEDIUM"|"LONG"}],
    "epfo_match": [{"resume_company": "string", "verified_company": "string", "date_match": boolean}],
    "only_in_resume": [{"company": "string", "flag": "string"}],
    "verdict": "MATCH"|"PARTIAL"|"MISMATCH"|"CLEAN"|"GAPS_FOUND",
    "verdict_reason": "string",
    "red_flags": ["string"]
  }
}`,
 
  offer_letter: `{
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
}`,
 
  offer_salary: `{
  "type": "offer_salary",
  "offer_check": {
    "overall_score": number,
    "verdict": "AUTHENTIC"|"SUSPICIOUS"|"FAKE",
    "verdict_reason": "string",
    "checks": {
      "company_legitimacy": {"score": number, "status": "PASS"|"WARN"|"FAIL", "finding": "string"},
      "ctc_format": {"score": number, "status": "PASS"|"WARN"|"FAIL", "finding": "string"},
      "designation_validity": {"score": number, "status": "PASS"|"WARN"|"FAIL", "finding": "string"},
      "dates_logic": {"score": number, "status": "PASS"|"WARN"|"FAIL", "finding": "string"},
      "language_quality": {"score": number, "status": "PASS"|"WARN"|"FAIL", "finding": "string"},
      "red_flags_check": {"score": number, "status": "PASS"|"WARN"|"FAIL", "finding": "string"}
    },
    "red_flags_list": ["string"],
    "positive_signals": ["string"]
  },
  "salary_check": {
    "overall_score": number,
    "verdict": "GENUINE"|"SUSPICIOUS"|"FAKE",
    "verdict_reason": "string",
    "rule_checks": [{"name": "string", "status": "pass"|"fail"|"neutral", "finding": "string"}],
    "red_flags": ["string"],
    "ctc_cross_check": "string"
  }
}`,
 
  salary_slip: `{
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
}`,
 
  salary_resume: `{
  "type": "salary_resume",
  "salary_check": {
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
  },
  "resume_check": {
    "mode": "gap_analysis",
    "companies": [{"company": "string", "from": "YYYY-MM", "to": "YYYY-MM|Present", "role": "string"}],
    "gaps": [{"from_company": "string", "from_date": "YYYY-MM", "to_company": "string", "to_date": "YYYY-MM", "gap_months": number, "severity": "SHORT"|"MEDIUM"|"LONG"}],
    "total_gap_months": number,
    "verdict": "CLEAN"|"GAPS_FOUND",
    "red_flags": ["string"]
  }
}`,
 
  epfo_crosscheck: `{
  "type": "employment",
  "mode": "epfo_crosscheck",
  "matched": [{"resume_company": "string", "verified_company": "string", "date_match": boolean}],
  "only_in_resume": [{"company": "string", "flag": "string"}],
  "experience_summary": {"resume_total_years": number, "epfo_total_years": number, "inflation_years": number},
  "verdict": "MATCH"|"PARTIAL"|"MISMATCH",
  "verdict_reason": "string",
  "red_flags": ["string"]
}`,
 
  epfo_only: `{
  "type": "employment",
  "mode": "epfo_summary",
  "companies": [{"company": "string", "from": "YYYY-MM", "to": "YYYY-MM|Present", "uan": "string|null"}],
  "total_verified_years": number,
  "verdict": "COMPLETE"|"GAPS_FOUND",
  "verdict_reason": "string",
  "red_flags": ["string"]
}`,
 
  resume_only: `{
  "type": "employment",
  "mode": "gap_analysis",
  "companies": [{"company": "string", "from": "YYYY-MM", "to": "YYYY-MM|Present", "role": "string"}],
  "gaps": [{"from_company": "string", "from_date": "YYYY-MM", "to_company": "string", "to_date": "YYYY-MM", "gap_months": number, "severity": "SHORT"|"MEDIUM"|"LONG", "label": "string"}],
  "total_gap_months": number,
  "verdict": "CLEAN"|"GAPS_FOUND",
  "verdict_reason": "string",
  "red_flags": ["string"]
}`,
 
  unknown_docs: `{
  "type": "unknown",
  "detected_content": "string",
  "findings": ["string"],
  "verdict": "string",
  "red_flags": ["string"]
}`,
};
 
// ── Shared prompt builder ─────────────────────────────────────────────────────
function buildUnifiedPrompt(documents, claimedCTC, mode, userMessage) {
  // FIX 1: Per-doc safe type detection
  for (const doc of documents) {
    try {
      if (!doc.detectedType || doc.detectedType === "unknown") {
        doc.detectedType = detectDocType(doc.text || "", doc.fileName || "");
      }
    } catch (e) {
      console.error(`detectDocType failed for ${doc.fileName}:`, e);
      doc.detectedType = "unknown";
    }
  }
 
  // Determine analysis mode
  const analysisMode = mode === "moonlighting"
    ? "moonlighting"
    : determineAnalysisMode(documents);
 
  // FIX 5: Auto-extract CTC from docs if not provided by user
  let effectiveCTC = claimedCTC || null;
  if (!effectiveCTC) {
    for (const doc of documents) {
      const found = extractCTCFromText(doc.text || "");
      if (found) { effectiveCTC = found; break; }
    }
  }
 
  // Pre-compute validators with try/catch per validator
  let validatorCtx = "";
 
  try {
    const offerDoc = documents.find(d => d.detectedType === "offer_letter");
    if (offerDoc) {
      const cin   = validateCIN(offerDoc.text || "");
      const gst   = validateGST(offerDoc.text || "");
      const isMNC = /accenture|opentext|cognizant|capgemini|oracle|infosys|wipro|tcs\b|ibm\b|deloitte/i.test(offerDoc.text || "");
      const year  = ((offerDoc.text || "").match(/\b(20[012]\d|19[89]\d)\b/) || [])[1];
      validatorCtx += `OFFER PRE-COMPUTED (do NOT re-validate):
CIN: ${cin.found ? `${cin.raw} → ${cin.verdict}` : "NOT FOUND"}
GST: ${gst.found ? `${gst.raw} → ${gst.verdict}` : "NOT FOUND"}
IS_MNC: ${isMNC} | YEAR: ${year || "unknown"}
NOTE: Pre-2017 letters → GST absence is NORMAL. MNC → CIN absence is NORMAL. INVALID_CHECKSUM = FAKE.\n\n`;
    }
  } catch (e) { console.error("Offer validator error:", e); }
 
  try {
    const salDoc = documents.find(d => d.detectedType === "salary_slip");
    if (salDoc) {
      const gst        = validateGST(salDoc.text || "");
      const gross      = extractAmt(salDoc.text, "Gross Earnings", "Total Earnings", "Gross Salary", "Gross");
      const basic      = extractAmt(salDoc.text, "Basic Salary", "Basic Pay", "Basic");
      const pf         = extractAmt(salDoc.text, "Provident Fund", "PF", "EPF", "Employee PF");
      const deductions = extractAmt(salDoc.text, "Total Deductions", "Net Deductions");
      const net        = extractAmt(salDoc.text, "Net Pay", "Net Salary", "Take Home");
      const pfExp      = basic ? basic * 0.12 : null;
      const pfOk       = pf && pfExp ? (pf === 1800 || Math.abs(pf - pfExp) <= Math.max(50, pfExp * 0.01)) : null;
      const netOk      = gross && deductions && net ? Math.abs(net - (gross - deductions)) <= Math.max(10, gross * 0.002) : null;
      validatorCtx += `SALARY PRE-COMPUTED:
GST: ${gst.found ? `${gst.raw} → ${gst.verdict}` : "NOT FOUND"}
Gross=\u20b9${gross||"?"}, Basic=\u20b9${basic||"?"}, PF=\u20b9${pf||"?"} (expected 12%=\u20b9${pfExp?.toFixed(0)||"?"}) \u2192 PF_VALID:${pfOk===null?"?":pfOk}
Net math: ${netOk===null?"?":netOk?"VALID":"MISMATCH \u2014 possible tampering"}\n`;
      if (effectiveCTC && gross) {
        const employerPF = basic ? Math.min(basic * 0.12, 1800) : 1800;
        const gratuity   = basic ? basic * 0.0481 : 0;
        const derivedAnn = Math.round((gross + employerPF + gratuity) * 12);
        const claimed    = effectiveCTC < 500 ? effectiveCTC * 100000 : effectiveCTC;
        const gap        = claimed - derivedAnn;
        const inflatePct = Math.round((gap / derivedAnn) * 1000) / 10;
        validatorCtx += `CTC: Derived=\u20b9${(derivedAnn/100000).toFixed(2)}L, Claimed=\u20b9${(claimed/100000).toFixed(2)}L, Gap=${inflatePct>0?'+':''}${inflatePct}% \u2192 ${Math.abs(inflatePct)<=15?"NORMAL":inflatePct>30?"HIGHLY_INFLATED":"INFLATED"}\n`;
      }
    }
  } catch (e) { console.error("Salary validator error:", e); }
 
  const jsonSchema = SCHEMAS[analysisMode] || SCHEMAS.unknown_docs;
 
  // FIX 4: Explicit no-echo instruction in system prompt
  const systemPrompt = `You are ParamkaraCorp-HR, an expert Indian HR fraud detection AI.
Analysis mode: ${analysisMode.toUpperCase()}
${validatorCtx}
STRICT RULES:
- Return ONLY valid JSON matching the schema below. Zero prose, zero markdown, zero preamble.
- NEVER repeat, quote, or summarise text from the input documents. JSON output only.
- Use pre-computed values exactly as given; do not re-derive them.
- Be conservative: flag suspicious items as WARN not FAIL unless clearly fake.
- Indian context: PF ceiling is \u20b91800/month (basic cap \u20b915000), GST introduced July 2017.
- Score 0-100: 85+ = PASS, 60-84 = WARN, <60 = FAIL.
- If a section has no supporting document (e.g. no EPFO doc for epfo_match), return empty array [].
 
REQUIRED JSON SCHEMA:
${jsonSchema}`;
 
  // Build document sections safely
  const docSections = documents.map((d, i) => {
    try {
      const label = (d.detectedType || "UNKNOWN").toUpperCase().replace(/_/g, " ");
      if (d.base64Image) return `DOCUMENT ${i+1} [${label}] (${d.fileName || "image"}): [IMAGE \u2014 analyze visually]`;
      return `--- DOCUMENT ${i+1}: ${d.fileName || "unnamed"} [${label}] ---\n${(d.text || "").slice(0, 4000)}`;
    } catch (e) {
      console.error(`Doc section build error for doc ${i}:`, e);
      return `--- DOCUMENT ${i+1}: [error reading document] ---`;
    }
  }).join("\n\n");
 
  const userPrompt = `${docSections}${userMessage ? `\n\nAdditional HR context: ${userMessage}` : ""}`;
 
  return { systemPrompt, userPrompt, analysisMode, effectiveCTC };
}
 
// ── Conversational fallback — ONLY when no documents sent ────────────────────
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
 
    // FIX 2: Conversation only when genuinely no documents — no fall-through from doc routing
    if (!documents || documents.length === 0) {
      const result = await conversationalReply(message || "Hello", history || []);
      return {
        statusCode: 200,
        headers: { ...CORS, "Content-Type": "application/json" },
        body: JSON.stringify(result),
      };
    }
 
    // FIX 1: Safe server-side type detection — per-doc try/catch
    for (const doc of documents) {
      try {
        if (!doc.detectedType || doc.detectedType === "unknown") {
          doc.detectedType = detectDocType(doc.text || "", doc.fileName || "");
        }
      } catch (e) {
        console.error(`Type detection failed for ${doc.fileName}:`, e);
        doc.detectedType = "unknown";
      }
    }
 
    // Build unified prompt
    let promptResult;
    try {
      promptResult = buildUnifiedPrompt(documents, claimedCTC, mode, message);
    } catch (e) {
      console.error("buildUnifiedPrompt fatal error:", e.stack || e);
      return {
        statusCode: 500,
        headers: { ...CORS, "Content-Type": "application/json" },
        body: JSON.stringify({ error: `Prompt build failed: ${e.message}` }),
      };
    }
 
    const { systemPrompt, userPrompt, analysisMode, effectiveCTC } = promptResult;
    const hasImage = documents.some(d => d.base64Image);
 
    let result;
    if (hasImage) {
      const contentBlocks = [];
      for (const doc of documents) {
        try {
          if (doc.base64Image) {
            contentBlocks.push({
              type: "image_url",
              image_url: { url: `data:${doc.imageMediaType};base64,${doc.base64Image}` },
            });
            contentBlocks.push({ type: "text", text: `Above image: ${doc.fileName}` });
          } else if (doc.text) {
            contentBlocks.push({ type: "text", text: `--- ${doc.fileName} ---\n${doc.text.slice(0, 3000)}` });
          }
        } catch (e) {
          console.error(`Vision block error for ${doc.fileName}:`, e);
        }
      }
      contentBlocks.push({ type: "text", text: `\nReturn JSON only per schema:\n${systemPrompt.split("REQUIRED JSON SCHEMA:")[1] || ""}` });
 
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
        // Fallback to text model
        result = await groq([
          { role: "system", content: systemPrompt },
          { role: "user", content: "[Image uploaded \u2014 analyze from context and file name]\n" + userPrompt },
        ], true, 2000);
      } else {
        const vd = await visionRes.json();
        const ct = vd.choices?.[0]?.message?.content ?? "{}";
        const match = ct.match(/\{[\s\S]*\}/);
        result = match ? JSON.parse(match[0]) : JSON.parse(ct);
      }
    } else {
      result = await groq([
        { role: "system", content: systemPrompt },
        { role: "user",   content: userPrompt },
      ], true, 2500);
    }
 
    // Ensure type is set
    if (!result.type) result.type = analysisMode;
 
    // Attach server-computed CTC inflation (overrides AI guess)
    if (effectiveCTC) {
      const salDoc = documents.find(d => d.detectedType === "salary_slip");
      if (salDoc) {
        try {
          const gross  = extractAmt(salDoc.text, "Gross Earnings", "Total Earnings", "Gross Salary", "Gross");
          const basic  = extractAmt(salDoc.text, "Basic Salary", "Basic Pay", "Basic");
          if (gross) {
            const employerPF = basic ? Math.min(basic * 0.12, 1800) : 1800;
            const gratuity   = basic ? basic * 0.0481 : 0;
            const derivedAnn = Math.round((gross + employerPF + gratuity) * 12);
            const claimed    = effectiveCTC < 500 ? effectiveCTC * 100000 : effectiveCTC;
            const gap        = claimed - derivedAnn;
            const inflatePct = Math.round((gap / derivedAnn) * 1000) / 10;
            const ctcObj = {
              derived_annual_ctc: derivedAnn,
              claimed_ctc_annual: claimed,
              inflation_gap:      Math.round(gap),
              inflation_percent:  inflatePct,
              verdict:            Math.abs(inflatePct) <= 15 ? "NORMAL" : inflatePct > 30 ? "HIGHLY_INFLATED" : "INFLATED",
            };
            // Attach to correct nesting
            if (result.salary_check) result.salary_check.ctc_inflation = ctcObj;
            else if (result.type === "salary_slip" || result.type === "salary_resume") result.ctc_inflation = ctcObj;
          }
        } catch (e) { console.error("CTC post-computation error:", e); }
      }
    }
 
    // FIX 5: Tell client CTC was auto-extracted so it can skip the modal
    if (effectiveCTC && !claimedCTC) {
      result._ctc_auto_extracted = effectiveCTC;
    }
 
    return {
      statusCode: 200,
      headers: { ...CORS, "Content-Type": "application/json" },
      body: JSON.stringify(result),
    };
 
  } catch (err) {
    console.error("Agent fatal error:", err.stack || err);
    return {
      statusCode: 500,
      headers: { ...CORS, "Content-Type": "application/json" },
      body: JSON.stringify({ error: err.message || "Internal server error" }),
    };
  }
};
 
