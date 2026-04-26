// ParamkaraCorp-HR Agent — v2 (fixed skill extraction + multi-rank)
// Fixes: skill years from date ranges, full candidate skill tables, longer text slices

const GROQ_API_KEY = process.env.GROQ_API_KEY;
const MODEL        = "llama-3.3-70b-versatile";
const VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct";

const CORS = {
  "Access-Control-Allow-Origin":  "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

async function groq(messages, json = false, maxTokens = 4000) {
  const body = { model: MODEL, messages, temperature: 0.05, max_tokens: maxTokens };
  if (json) body.response_format = { type: "json_object" };
  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: { Authorization: `Bearer ${GROQ_API_KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`Groq ${res.status}: ${await res.text()}`);
  const data = await res.json();
  const content = data.choices?.[0]?.message?.content ?? "";
  if (json) {
    const m = content.match(/\{[\s\S]*\}/);
    try { return m ? JSON.parse(m[0]) : JSON.parse(content); }
    catch { throw new Error("JSON parse failed: " + content.slice(0, 300)); }
  }
  return content;
}

// ── Validators ────────────────────────────────────────────────────────────────
function validateGST(text) {
  if (!text) return { found: false };
  const m = text.match(/\b([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z])\b/);
  if (!m) return { found: false };
  const gst = m[1], CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  let factor = 1, sum = 0;
  for (let i = 0; i < 14; i++) {
    let d = factor * CHARS.indexOf(gst[i]);
    d = Math.floor(d / 36) + (d % 36); sum += d; factor = factor === 1 ? 3 : 1;
  }
  const exp = CHARS[(36 - (sum % 36)) % 36];
  return { found: true, raw: gst, valid: gst[14] === exp, verdict: gst[14] === exp ? "VALID" : "INVALID_CHECKSUM" };
}

function validateCIN(text) {
  if (!text) return { found: false };
  const m = text.match(/\b([LUlu][0-9]{5}[A-Za-z]{2}[0-9]{4}[A-Za-z]{3}[0-9]{6})\b/);
  if (!m) return { found: false };
  const cin = m[1].toUpperCase();
  const STATES = new Set(["AN","AP","AR","AS","BR","CH","CG","DD","DL","DN","GA","GJ","HP","HR","JH","JK","KA","KL","LA","LD","MH","ML","MN","MP","MZ","NL","OD","PB","PY","RJ","SK","TN","TR","TS","UK","UP","WB"]);
  const TYPES  = new Set(["PLC","PTC","GOI","SGC","FLC","FTC","NPL","ULL","ULT","GAP","GAT"]);
  const valid  = STATES.has(cin.slice(6,8)) && parseInt(cin.slice(8,12)) >= 1850 && TYPES.has(cin.slice(12,15));
  return { found: true, raw: cin, valid, verdict: valid ? "VALID" : "INVALID_FORMAT" };
}

function extractAmt(text, ...labels) {
  for (const label of labels) {
    const esc = label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s*");
    const m = text?.match(new RegExp(`${esc}\\s*[:\\-]?\\s*(?:INR|₹|Rs\\.?)?\\s*([0-9][0-9,]*)`, "i"));
    if (m?.[1]) { const v = parseFloat(m[1].replace(/,/g,"")); if (!isNaN(v) && v > 0) return v; }
  }
  return null;
}

function extractCTC(text) {
  if (!text) return null;
  for (const pat of [
    /(?:expected\s+ctc|ctc\s+offered|ctc|cost\s+to\s+company)\s*[:\-]?\s*(?:INR|₹|Rs\.?)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:lpa|l\.p\.a|lakhs?(?:\s*per\s*annum)?|lac)/i,
    /(?:expected\s+ctc|ctc\s+offered|ctc|cost\s+to\s+company)\s*[:\-]?\s*(?:INR|₹|Rs\.?)?\s*([0-9][0-9,]+)/i,
  ]) {
    const m = text.match(pat);
    if (m?.[1]) { const v = parseFloat(m[1].replace(/,/g,"")); if (!isNaN(v) && v > 0) return v < 500 ? v * 100000 : v; }
  }
  return null;
}

// ── Doc type detection ────────────────────────────────────────────────────────
function detectType(text, fileName) {
  const t = (text || '').toLowerCase();
  const f = (fileName || '').toLowerCase();
 
  // ── SALARY — require STRONG explicit signals only ─────────────────
  // A real salary slip ALWAYS has one of these. A resume NEVER does.
  const isSalary =
    t.includes('salary slip') ||
    t.includes('payslip') ||
    t.includes('pay slip') ||
    t.includes('gross earnings') ||
    t.includes('gross salary') ||
    t.includes('net pay') ||
    t.includes('net salary') ||
    t.includes('take home pay') ||
    (t.includes('month') && t.includes('basic') && t.includes('hra') &&
     t.includes('net') && (t.includes('₹') || t.includes('inr') || t.includes('rs.')));
 
  // ── RESUME — strong structural patterns ───────────────────────────
  // Real resumes always have these combinations
  const isResume =
    t.includes('curriculum vitae') ||
    t.includes('professional summary') ||
    t.includes('career objective') ||
    t.includes('work experience') ||
    (t.includes('experience') && t.includes('education') && t.includes('skills')) ||
    (t.includes('objective') && t.includes('skills') && t.includes('experience'));
 
  // ── JD — specific JD language ─────────────────────────────────────
  const isJD =
    t.includes('job description') ||
    t.includes('we are looking for') ||
    t.includes('key responsibilities') ||
    t.includes('role overview') ||
    t.includes('mandatory skills') ||
    t.includes('good to have') ||
    t.includes('years of experience required') ||
    (t.includes('responsibilities') && t.includes('requirements') && !isResume);
 
  // ── OFFER LETTER ──────────────────────────────────────────────────
  const isOffer =
    t.includes('offer letter') ||
    t.includes('we are pleased to offer') ||
    t.includes('joining date') ||
    t.includes('date of joining') ||
    (t.includes('cost to company') && t.includes('designation') && !isResume);
 
  // ── EPFO / Service history ────────────────────────────────────────
  const isEPFO =
    t.includes('epfo') ||
    t.includes('service history') ||
    t.includes('uan') ||
    (t.includes('passbook') && t.includes('pf'));
 
  // ── BANK ──────────────────────────────────────────────────────────
  const isBank =
    t.includes('bank statement') ||
    t.includes('account statement') ||
    t.includes('account number') ||
    (t.includes('transaction') && t.includes('closing balance'));
 
  // ── Priority order ────────────────────────────────────────────────
  // Resume vs Salary can conflict — resume wins if both detected
  // because salary signals are now much stricter
  if (isOffer)  return 'offer';
  if (isEPFO)   return 'epfo';
  if (isBank)   return 'bank';
  if (isJD)     return 'jd';
  if (isResume) return 'resume';   // resume BEFORE salary
  if (isSalary) return 'salary';
 
  // ── Filename as last fallback only ────────────────────────────────
  if (f.includes('resume') || f.includes('_cv') || f.startsWith('cv'))  return 'resume';
  if (f.includes('jd') || f.includes('job'))                             return 'jd';
  if (f.includes('offer') || f.includes('appointment'))                  return 'offer';
  if (f.includes('salary') || f.includes('payslip'))                     return 'salary';
  if (f.includes('epfo') || f.includes('uan'))                           return 'epfo';
  if (f.includes('bank') || f.includes('statement'))                     return 'bank';
 
  return 'unknown';
}

// ── Route to analysis mode ────────────────────────────────────────────────────
function routeMode(documents, modeHint) {
  if (modeHint === "moonlighting") return "moonlighting";
  const types = documents.map(d => d.detectedType || "unknown");
  const has = t => types.includes(t);
  const count = t => types.filter(x => x === t).length;

  const hasJD      = has("jd");
  const numResumes = count("resume");
  const hasOffer   = has("offer");
  const hasSalary  = has("salary");
  const hasEPFO    = has("epfo");
  const hasBank    = has("bank");

  if (hasJD && numResumes >= 2) return "multi_rank";
  if (hasJD && numResumes === 1) return "jd_resume";
  if (hasJD) return "jd_only";
  if (hasOffer && hasSalary) return "offer_salary";
  if (hasOffer) return "offer_letter";
  if ((hasSalary || hasBank) && numResumes >= 1) return "salary_resume";
  if (hasSalary || hasBank) return "salary_slip";
  if (numResumes >= 1 && hasEPFO) return "epfo_crosscheck";
  if (hasEPFO) return "epfo_only";
  if (numResumes >= 2) return "multi_resume_nojd";  // resumes only, no JD
  if (numResumes >= 1) return "resume_only";
  return "unknown_docs";
}

// ── Build system prompts ──────────────────────────────────────────────────────
function buildPrompt(documents, claimedCTC, modeHint) {
  for (const doc of documents) {
    if (!doc.detectedType || doc.detectedType === "unknown")
      doc.detectedType = detectType(doc.text || "", doc.fileName || "");
  }

  const analysisMode = routeMode(documents, modeHint);
  let effectiveCTC = claimedCTC || null;
  if (!effectiveCTC) {
    for (const doc of documents) { const c = extractCTC(doc.text||""); if (c) { effectiveCTC = c; break; } }
  }

  let validatorCtx = "";

  // Offer validators
  const offerDoc = documents.find(d => d.detectedType === "offer");
  if (offerDoc) {
    const cin = validateCIN(offerDoc.text||""), gst = validateGST(offerDoc.text||"");
    const isMNC = /accenture|cognizant|capgemini|oracle|infosys|wipro|tcs\b|ibm\b|deloitte/i.test(offerDoc.text||"");
    const year = ((offerDoc.text||"").match(/\b(20[012]\d|19[89]\d)\b/)||[])[1];
    validatorCtx += `OFFER VALIDATORS:\nCIN: ${cin.found?`${cin.raw}→${cin.verdict}`:"NOT FOUND"}\nGST: ${gst.found?`${gst.raw}→${gst.verdict}`:"NOT FOUND"}\nMNC:${isMNC} YEAR:${year||"?"}\nNOTE: Pre-2017→GST absence normal. MNC→CIN absence normal.\n\n`;
  }

  // Salary validators
  const salDoc = documents.find(d => d.detectedType === "salary" || d.detectedType === "bank");
  if (salDoc) {
    const gross = extractAmt(salDoc.text,"Gross Earnings","Total Earnings","Gross Salary","Gross");
    const basic = extractAmt(salDoc.text,"Basic Salary","Basic Pay","Basic");
    const pf    = extractAmt(salDoc.text,"Provident Fund","PF","EPF","Employee PF");
    const net   = extractAmt(salDoc.text,"Net Pay","Net Salary","Take Home");
    const ded   = extractAmt(salDoc.text,"Total Deductions","Net Deductions");
    const pfExp = basic ? basic * 0.12 : null;
    const pfOk  = pf && pfExp ? (pf === 1800 || Math.abs(pf-pfExp) <= Math.max(50,pfExp*0.01)) : null;
    const netOk = gross&&ded&&net ? Math.abs(net-(gross-ded)) <= Math.max(10,gross*0.002) : null;
    validatorCtx += `SALARY VALIDATORS:\nGross=₹${gross||"?"} Basic=₹${basic||"?"} PF=₹${pf||"?"} (exp=₹${pfExp?.toFixed(0)||"?"}) PF_OK:${pfOk??'?'}\nNet math: ${netOk==null?'?':netOk?'VALID':'MISMATCH—tampering possible'}\n`;
    if (effectiveCTC && gross) {
      const empPF = basic ? Math.min(basic*0.12,1800) : 1800;
      const grat  = basic ? basic*0.0481 : 0;
      const derived = Math.round((gross+empPF+grat)*12);
      const claimed = effectiveCTC < 500 ? effectiveCTC*100000 : effectiveCTC;
      const pct   = Math.round(((claimed-derived)/derived)*1000)/10;
      validatorCtx += `CTC: Derived=₹${(derived/100000).toFixed(2)}L Claimed=₹${(claimed/100000).toFixed(2)}L ${pct>0?'+':''}${pct}% → ${Math.abs(pct)<=15?"NORMAL":pct>30?"HIGHLY_INFLATED":"INFLATED"}\n`;
    }
  }

  // ── Build doc sections — INCREASED slice to 7000 for resumes ──────────────
  const docSections = documents.map((d, i) => {
    const label = (d.detectedType||"unknown").toUpperCase();
    if (d.base64Image) return `DOC ${i+1} [${label}] (${d.fileName}): [IMAGE]`;
    // Give resumes more context so skills + dates can be fully parsed
    const limit = d.detectedType === "resume" ? 7000
            : d.detectedType === "epfo"   ? 3000   // PF docs are long, keep short
            : 4000;
    return `--- DOC ${i+1}: ${d.fileName} [${label}] ---\n${(d.text||"").slice(0, limit)}`;
  }).join("\n\n");

  const schema = SCHEMAS[analysisMode] || SCHEMAS.unknown_docs;

  // ── FIXED: rules no longer kill skill arrays; added explicit years extraction instructions
  const system = `You are ParamkaraCorp-HR, expert Indian HR fraud detection and recruitment AI.
Analysis mode: ${analysisMode.toUpperCase()}
${validatorCtx}
STRICT RULES:
1. Output ONLY valid JSON. Zero prose, zero markdown, zero text outside JSON.
2. verdict_reason, recommendation_reason: ONE short sentence (max 20 words).
3. strengths/gaps arrays: max 4 items, max 8 words each.
4. skill names: use exact name from JD or resume, keep concise.
5. NEVER reproduce document text verbatim.
6. Use pre-computed validator values exactly as given.
7. Indian PF ceiling: ₹1800/month (basic cap ₹15000). GST since July 2017.
8. Scores 0-100: 80+=SHORTLIST, 55-79=MAYBE, <55=REJECT.

HOW TO COMPUTE actual_years PER SKILL:
- Read ALL company entries in the resume: company name, start date, end date (or "Present").
- For each role, identify which skills/tools were used based on job title and description.
- Compute duration: (end_year - start_year) + (end_month - start_month)/12. Round to 1 decimal.
- Sum durations across all roles where that skill was used → that is actual_years.
- If a skill is listed in Skills section but no role mentions it, use 0.5 as a conservative estimate.
- If resume has no dates, use total_experience_years divided by skill count as fallback.
- DO NOT leave actual_years as 0 if the candidate clearly has the skill — estimate from context.
- required_years: extract from JD (e.g. "5+ years of Python" → 5). null if JD doesn't specify.
- matched: true if actual_years >= required_years (or actual_years > 0 when required_years is null).

REQUIRED JSON SCHEMA:
${schema}`;

  return { system, userPrompt: docSections, analysisMode, effectiveCTC };
}

// ── Schemas ───────────────────────────────────────────────────────────────────
const SCHEMAS = {

  // ── JD vs Single Resume ───────────────────────────────────────────────────
  jd_resume: `{
  "type": "jd_resume",
  "candidate": {
    "name": "string",
    "total_experience_years": number,
    "current_role": "string",
    "current_company": "string"
  },
  "skill_match": [
    {
      "skill": "string",
      "required_years": number|null,
      "actual_years": number,
      "matched": boolean,
      "used_at": "company1, company2"
    }
  ],
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

  // ── Multi-candidate ranking (JD + 2+ resumes) ────────────────────────────
  multi_rank: `{
  "type": "multi_rank",
  "role": "string",
  "jd_skills": ["skill1", "skill2"],
  "candidates": [
    {
      "name": "string",
      "total_experience_years": number,
      "current_role": "string",
      "match_percentage": number,
      "mandatory_met": number,
      "mandatory_total": number,
      "recommendation": "SHORTLIST"|"MAYBE"|"REJECT",
      "recommendation_reason": "string",
      "skill_match": [
        {
          "skill": "string",
          "required_years": number|null,
          "actual_years": number,
          "matched": boolean,
          "used_at": "company1, company2"
        }
      ]
    }
  ]
}`,

  jd_only: `{
  "type": "jd_only",
  "jd_summary": {"role": "string", "company": "string|null", "ctc_range": "string|null"},
  "required_skills": [{"skill": "string", "mandatory": boolean, "years": number|null}]
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
    "red_flags_check": {"score": number, "status": "PASS"|"WARN"|"FAIL", "finding": "string"}
  },
  "red_flags_list": ["string"],
  "positive_signals": ["string"]
}`,

  offer_salary: `{
  "type": "offer_salary",
  "offer_check": {
    "overall_score": number, "verdict": "AUTHENTIC"|"SUSPICIOUS"|"FAKE", "verdict_reason": "string",
    "checks": {
      "company_legitimacy": {"score": number, "status": "PASS"|"WARN"|"FAIL", "finding": "string"},
      "ctc_format": {"score": number, "status": "PASS"|"WARN"|"FAIL", "finding": "string"},
      "designation_validity": {"score": number, "status": "PASS"|"WARN"|"FAIL", "finding": "string"},
      "dates_logic": {"score": number, "status": "PASS"|"WARN"|"FAIL", "finding": "string"},
      "language_quality": {"score": number, "status": "PASS"|"WARN"|"FAIL", "finding": "string"},
      "red_flags_check": {"score": number, "status": "PASS"|"WARN"|"FAIL", "finding": "string"}
    },
    "red_flags_list": ["string"], "positive_signals": ["string"]
  },
  "salary_check": {
    "overall_score": number, "verdict": "GENUINE"|"SUSPICIOUS"|"FAKE", "verdict_reason": "string",
    "rule_checks": [{"name": "string", "status": "pass"|"fail"|"neutral", "finding": "string"}],
    "red_flags": ["string"],
    "ctc_inflation": {"derived_annual_ctc": number|null, "claimed_ctc_annual": number|null, "inflation_percent": number|null, "verdict": "NORMAL"|"INFLATED"|"HIGHLY_INFLATED"|"NO_CLAIM_TO_COMPARE"}
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
  "ctc_inflation": {"derived_annual_ctc": number|null, "claimed_ctc_annual": number|null, "inflation_percent": number|null, "verdict": "NORMAL"|"INFLATED"|"HIGHLY_INFLATED"|"NO_CLAIM_TO_COMPARE"}
}`,

  salary_resume: `{
  "type": "salary_resume",
  "salary_check": {
    "overall_score": number, "verdict": "GENUINE"|"SUSPICIOUS"|"FAKE", "verdict_reason": "string",
    "rule_checks": [{"name": "string", "status": "pass"|"fail"|"neutral", "finding": "string"}],
    "ai_reasoning": "string", "red_flags": ["string"],
    "ctc_inflation": {"derived_annual_ctc": number|null, "claimed_ctc_annual": number|null, "inflation_percent": number|null, "verdict": "NORMAL"|"INFLATED"|"HIGHLY_INFLATED"|"NO_CLAIM_TO_COMPARE"}
  },
  "resume_check": {
    "mode": "gap_analysis",
    "companies": [{"company": "string", "from": "YYYY-MM", "to": "YYYY-MM|Present", "role": "string"}],
    "gaps": [{"from_company": "string", "to_company": "string", "gap_months": number, "severity": "SHORT"|"MEDIUM"|"LONG", "label": "string"}],
    "total_gap_months": number, "verdict": "CLEAN"|"GAPS_FOUND", "red_flags": ["string"]
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
  "companies": [{"company": "string", "from": "YYYY-MM", "to": "YYYY-MM|Present"}],
  "total_verified_years": number,
  "verdict": "COMPLETE"|"GAPS_FOUND",
  "verdict_reason": "string",
  "red_flags": ["string"]
}`,

  resume_only: `{
  "type": "employment",
  "mode": "gap_analysis",
  "companies": [{"company": "string", "from": "YYYY-MM", "to": "YYYY-MM|Present", "role": "string"}],
  "gaps": [{"from_company": "string", "to_company": "string", "gap_months": number, "severity": "SHORT"|"MEDIUM"|"LONG", "label": "string"}],
  "total_gap_months": number,
  "verdict": "CLEAN"|"GAPS_FOUND",
  "verdict_reason": "string",
  "red_flags": ["string"]
}`,

  multi_resume_nojd: `{
  "type": "multi_resume_nojd",
  "message": "string",
  "candidates": [
    {"name": "string", "total_experience_years": number, "current_role": "string",
     "companies": [{"company": "string", "from": "YYYY-MM", "to": "YYYY-MM|Present", "role": "string"}],
     "skills_extracted": [{"skill": "string", "years": number}],
     "verdict": "CLEAN"|"GAPS_FOUND", "total_gap_months": number
    }
  ]
}`,

  moonlighting: `{
  "type": "moonlighting",
  "verdict": "LOW_RISK"|"MEDIUM_RISK"|"HIGH_RISK",
  "verdict_reason": "string",
  "signals": [{"signal": "string", "risk": "LOW"|"MEDIUM"|"HIGH"}],
  "recommendations": ["string"]
}`,

  unknown_docs: `{
  "type": "unknown",
  "detected_content": "string",
  "findings": ["string"],
  "verdict": "string",
  "red_flags": ["string"]
}`,
};

// ── Conversational fallback ───────────────────────────────────────────────────
async function chat(message, history) {
  const msgs = [
    { role: "system", content: `You are ParamkaraCorp-HR Assistant for Indian HR professionals. Help with: resume screening, JD matching, offer letter fraud, salary verification, EPFO checks. Be concise — max 3 sentences. Use bullet points only when listing.` },
    ...(history||[]).slice(-4),
    { role: "user", content: message },
  ];
  return { type: "conversation", reply: await groq(msgs, false, 300) };
}

// ── Main handler ──────────────────────────────────────────────────────────────
exports.handler = async (event) => {
  if (event.httpMethod === "OPTIONS") return { statusCode: 200, headers: CORS, body: "" };
  if (event.httpMethod !== "POST") return { statusCode: 405, headers: CORS, body: JSON.stringify({ error: "Method not allowed" }) };
  if (!GROQ_API_KEY) return { statusCode: 500, headers: { ...CORS, "Content-Type": "application/json" }, body: JSON.stringify({ error: "GROQ_API_KEY not set" }) };

  try {
    const body = JSON.parse(event.body || "{}");
    let { message, documents, history, claimedCTC, mode } = body;

    // Pure conversation — no documents
    if (!documents || documents.length === 0) {
      const r = await chat(message || "Hello", history);
      return { statusCode: 200, headers: { ...CORS, "Content-Type": "application/json" }, body: JSON.stringify(r) };
    }

    // Server-side type detection
    for (const doc of documents) {
      if (!doc.detectedType || doc.detectedType === "unknown")
        doc.detectedType = detectType(doc.text || "", doc.fileName || "");
    }

    const { system, userPrompt, analysisMode, effectiveCTC } = buildPrompt(documents, claimedCTC, mode);
    const hasImage = documents.some(d => d.base64Image);

    let result;

    if (hasImage) {
      const blocks = [];
      for (const doc of documents) {
        if (doc.base64Image) {
          blocks.push({ type: "image_url", image_url: { url: `data:${doc.imageMediaType};base64,${doc.base64Image}` } });
          blocks.push({ type: "text", text: `Image: ${doc.fileName}` });
        } else if (doc.text) {
          blocks.push({ type: "text", text: `--- ${doc.fileName} [${doc.detectedType}] ---\n${doc.text.slice(0,4000)}` });
        }
      }
      blocks.push({ type: "text", text: `\nReturn JSON per schema:\n${system.split("REQUIRED JSON SCHEMA:")[1]||""}` });

      const vRes = await fetch("https://api.groq.com/openai/v1/chat/completions", {
        method: "POST",
        headers: { Authorization: `Bearer ${GROQ_API_KEY}`, "Content-Type": "application/json" },
        body: JSON.stringify({ model: VISION_MODEL, messages: [{ role: "system", content: system }, { role: "user", content: blocks }], temperature: 0.05, max_tokens: 3000, response_format: { type: "json_object" } }),
      });
      if (!vRes.ok) {
        result = await groq([{ role: "system", content: system }, { role: "user", content: "[Image]\n" + userPrompt }], true, 3000);
      } else {
        const vd = await vRes.json();
        const ct = vd.choices?.[0]?.message?.content ?? "{}";
        const m = ct.match(/\{[\s\S]*\}/);
        result = m ? JSON.parse(m[0]) : JSON.parse(ct);
      }
    } else {
      // For multi_rank with many resumes, bump token limit so all candidates fit
      const isMulti = analysisMode === "multi_rank";
      const tokenLimit = isMulti ? 6000 : 4000;
      result = await groq([{ role: "system", content: system }, { role: "user", content: userPrompt }], true, tokenLimit);
    }

    if (!result.type) result.type = analysisMode;

    // Attach server-computed CTC inflation
    if (effectiveCTC) {
      const salDoc = documents.find(d => d.detectedType === "salary" || d.detectedType === "bank");
      if (salDoc) {
        const gross = extractAmt(salDoc.text,"Gross Earnings","Total Earnings","Gross Salary","Gross");
        const basic = extractAmt(salDoc.text,"Basic Salary","Basic Pay","Basic");
        if (gross) {
          const empPF = basic ? Math.min(basic*0.12,1800) : 1800;
          const grat  = basic ? basic*0.0481 : 0;
          const derived = Math.round((gross+empPF+grat)*12);
          const claimed = effectiveCTC < 500 ? effectiveCTC*100000 : effectiveCTC;
          const pct = Math.round(((claimed-derived)/derived)*1000)/10;
          const obj = { derived_annual_ctc:derived, claimed_ctc_annual:claimed, inflation_percent:pct, verdict: Math.abs(pct)<=15?"NORMAL":pct>30?"HIGHLY_INFLATED":"INFLATED" };
          if (result.salary_check) result.salary_check.ctc_inflation = obj;
          else result.ctc_inflation = obj;
        }
      }
    }

    if (effectiveCTC && !claimedCTC) result._ctc_auto_extracted = effectiveCTC;

    return { statusCode: 200, headers: { ...CORS, "Content-Type": "application/json" }, body: JSON.stringify(result) };

  } catch (err) {
    console.error("Fatal:", err.stack || err);
    return { statusCode: 500, headers: { ...CORS, "Content-Type": "application/json" }, body: JSON.stringify({ error: err.message || "Internal error" }) };
  }
};
