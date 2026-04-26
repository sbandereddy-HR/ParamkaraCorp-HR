// ParamkaraCorp-HR Agent — v5 (STRICT PIPELINE)
// Architecture: UPLOAD → DETECT → VALIDATE → QUEUE → ANALYZE (ONCE) → AGGREGATE → OUTPUT
// Single combined analysis call only. Resume-first enforced server-side.

const GROQ_API_KEY  = process.env.GROQ_API_KEY;
const MODEL         = "llama-3.3-70b-versatile";
const VISION_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct";

// ── Doc priority order (lower index = higher priority) ───────────────────────
const DOC_PRIORITY = ["resume", "jd", "salary", "epfo", "bank", "offer", "unknown"];
// ── Token budget per doc type ────────────────────────────────────────────────
const DOC_CHAR_LIMITS = { resume: 3000, jd: 1500, salary: 1500, epfo: 1500, bank: 1500, offer: 1500, unknown: 800 };
const MAX_DOCS = 4;

const CORS = {
  "Access-Control-Allow-Origin":  "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

// ── Groq caller ────────────────────────────────────────────────────────────────
async function groq(messages, json = false, maxTokens = 2500) {
  const body = { model: MODEL, messages, temperature: 0.05, max_tokens: maxTokens };
  if (json) body.response_format = { type: "json_object" };
  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: { Authorization: `Bearer ${GROQ_API_KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Groq ${res.status}: ${errText}`);
  }
  const data = await res.json();
  const content = data.choices?.[0]?.message?.content ?? "";
  if (json) {
    const m = content.match(/\{[\s\S]*\}/);
    try { return m ? JSON.parse(m[0]) : JSON.parse(content); }
    catch { throw new Error("JSON parse failed: " + content.slice(0, 300)); }
  }
  return content;
}

async function groqWithRetry(messages, json = false, maxTokens = 2500, retries = 2) {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try { return await groq(messages, json, maxTokens); }
    catch (e) {
      const retryable = e.message.includes("429") || e.message.includes("503") || e.message.includes("500");
      if (!retryable || attempt === retries) throw e;
      await new Promise(r => setTimeout(r, 2500 * (attempt + 1)));
    }
  }
}

// ── Validators ─────────────────────────────────────────────────────────────────
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

function parsePFAmounts(text) {
  const amounts = [];
  const rowPat = /Cont\.?\s+For\s+Due-\s*Month\s+(\d{6})\s+([0-9,]+)/gi;
  let m;
  while ((m = rowPat.exec(text)) !== null) {
    const v = parseFloat(m[2].replace(/,/g,""));
    if (!isNaN(v) && v > 0 && v <= 15000) amounts.push(v);
  }
  if (amounts.length > 0) return amounts;
  const namedPats = [
    /(?:Employee\s+(?:PF|Provident\s+Fund)|EE\s+PF|EPF\s+Employee)\s*[:\-]?\s*(?:₹|Rs\.?\s*)?([0-9,]+(?:\.\d{2})?)/gi,
    /(?:PF\s+Contribution|EPF\s+Amount|Monthly\s+PF)\s*[:\-]?\s*(?:₹|Rs\.?\s*)?([0-9,]+(?:\.\d{2})?)/gi,
  ];
  for (const p of namedPats) {
    while ((m = p.exec(text)) !== null) {
      const v = parseFloat(m[1].replace(/,/g,""));
      if (!isNaN(v) && v > 100 && v <= 15000) amounts.push(v);
    }
  }
  return amounts;
}

function parseBankCredits(text) {
  const amounts = [];
  const pats = [
    /(?:salary|sal|neft|imps|credit).*?([0-9,]{5,9}(?:\.\d{2})?)/gi,
    /([0-9,]{5,9}(?:\.\d{2})?)\s*(?:cr|credit)/gi,
  ];
  for (const p of pats) {
    let m;
    while ((m = p.exec(text)) !== null) {
      const v = parseFloat(m[1].replace(/,/g,""));
      if (!isNaN(v) && v >= 5000 && v <= 10000000) amounts.push(v);
    }
  }
  if (amounts.length === 0) {
    const all = [...text.matchAll(/\b([0-9]{5,7}(?:\.[0-9]{2})?)\b/g)]
      .map(m => parseFloat(m[1])).filter(n => n >= 10000 && n <= 500000);
    return all.slice(0, 12);
  }
  return amounts.slice(0, 12);
}

// ── Doc type detection ──────────────────────────────────────────────────────────
function detectType(text, fileName) {
  const t = (text || '').toLowerCase();
  const f = (fileName || '').toLowerCase();

  const isSalary =
    t.includes('salary slip') || t.includes('payslip') || t.includes('pay slip') ||
    t.includes('gross earnings') || t.includes('gross salary') ||
    t.includes('net pay') || t.includes('net salary') || t.includes('take home pay') ||
    (t.includes('month') && t.includes('basic') && t.includes('hra') &&
     t.includes('net') && (t.includes('₹') || t.includes('inr') || t.includes('rs.')));

  const isResume =
    t.includes('curriculum vitae') || t.includes('professional summary') ||
    t.includes('career objective') || t.includes('work experience') ||
    (t.includes('experience') && t.includes('education') && t.includes('skills')) ||
    (t.includes('objective') && t.includes('skills') && t.includes('experience'));

  const isJD =
    t.includes('job description') || t.includes('we are looking for') ||
    t.includes('key responsibilities') || t.includes('role overview') ||
    t.includes('mandatory skills') || t.includes('good to have') ||
    t.includes('years of experience required') ||
    (t.includes('responsibilities') && t.includes('requirements') && !isResume);

  const isOffer =
    t.includes('offer letter') || t.includes('we are pleased to offer') ||
    t.includes('joining date') || t.includes('date of joining') ||
    (t.includes('cost to company') && t.includes('designation') && !isResume);

  const isEPFO =
    t.includes('epfo') || t.includes('service history') || t.includes('uan') ||
    (t.includes('passbook') && t.includes('pf'));

  const isBank =
    t.includes('bank statement') || t.includes('account statement') ||
    t.includes('account number') ||
    (t.includes('transaction') && t.includes('closing balance'));

  if (isOffer)  return 'offer';
  if (isEPFO)   return 'epfo';
  if (isBank)   return 'bank';
  if (isJD)     return 'jd';
  if (isResume) return 'resume';
  if (isSalary) return 'salary';

  if (f.includes('resume') || f.includes('_cv') || f.startsWith('cv'))  return 'resume';
  if (f.includes('jd') || f.includes('job'))                             return 'jd';
  if (f.includes('offer') || f.includes('appointment'))                  return 'offer';
  if (f.includes('salary') || f.includes('payslip'))                     return 'salary';
  if (f.includes('epfo') || f.includes('uan'))                           return 'epfo';
  if (f.includes('bank') || f.includes('statement'))                     return 'bank';

  return 'unknown';
}

// ── Pipeline: apply doc limits ─────────────────────────────────────────────────
// Sort by priority, cap at MAX_DOCS, truncate text per type
function applyDocLimits(documents) {
  // Sort by priority
  const sorted = [...documents].sort((a, b) => {
    const pa = DOC_PRIORITY.indexOf(a.detectedType || 'unknown');
    const pb = DOC_PRIORITY.indexOf(b.detectedType || 'unknown');
    return (pa === -1 ? 99 : pa) - (pb === -1 ? 99 : pb);
  });

  // Cap at MAX_DOCS — drop lowest-priority first (already sorted)
  const limited = sorted.slice(0, MAX_DOCS);

  // Truncate each doc text
  return limited.map(doc => ({
    ...doc,
    text: doc.base64Image ? doc.text : (doc.text || '').slice(0, DOC_CHAR_LIMITS[doc.detectedType] || 800)
  }));
}

// ── Route to unified analysis mode ─────────────────────────────────────────────
function routeMode(documents) {
  const types = documents.map(d => d.detectedType || "unknown");
  const has   = t => types.includes(t);
  const count = t => types.filter(x => x === t).length;

  const hasJD      = has("jd");
  const numResumes = count("resume");
  const hasOffer   = has("offer");
  const hasSalary  = has("salary");
  const hasEPFO    = has("epfo");
  const hasBank    = has("bank");
  const hasCTC     = hasSalary || hasBank || hasEPFO;

  // Unified full analysis: resume + jd + any financial docs → comprehensive
  if (numResumes >= 1 && hasJD && hasCTC) return "full_analysis";
  if (numResumes >= 1 && hasJD && hasOffer) return "full_analysis";

  // EPFO cross-check
  if (hasEPFO && numResumes >= 1) return "epfo_crosscheck";

  // JD routing
  if (hasJD && numResumes >= 2) return "multi_rank";
  if (hasJD && numResumes === 1) return "jd_resume";
  if (hasJD) return "jd_only";

  // Offer letter
  if (hasOffer && hasSalary) return "offer_salary";
  if (hasOffer) return "offer_letter";

  // CTC/financial docs (with resume in payload)
  if (hasCTC && numResumes >= 1) return "ctc_verification";

  // Resume-only
  if (numResumes >= 2) return "multi_resume_nojd";
  if (numResumes >= 1) return "resume_only";

  return "unknown_docs";
}

// ── Build unified prompt ────────────────────────────────────────────────────────
function buildPrompt(documents, claimedCTC) {
  for (const doc of documents) {
    if (!doc.detectedType || doc.detectedType === "unknown")
      doc.detectedType = detectType(doc.text || "", doc.fileName || "");
  }

  const analysisMode = routeMode(documents);
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
  const salDoc = documents.find(d => d.detectedType === "salary");
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

  // PF validators
  const epfoDoc = documents.find(d => d.detectedType === "epfo");
  if (epfoDoc) {
    const pfAmounts = parsePFAmounts(epfoDoc.text || "");
    if (pfAmounts.length > 0) {
      const avgPF = pfAmounts.reduce((a,b)=>a+b,0) / pfAmounts.length;
      const derivedBasic = Math.round(avgPF / 0.12);
      const derivedMonthly = Math.round(derivedBasic / 0.4);
      const derivedAnnual = derivedMonthly * 12;
      validatorCtx += `\nPF VALIDATORS:\nPF amounts: ${pfAmounts.slice(0,6).join(', ')}${pfAmounts.length>6?'...':''}\nAvg PF/month=₹${Math.round(avgPF)} → Derived Basic=₹${derivedBasic} → Derived Annual≈₹${(derivedAnnual/100000).toFixed(2)}L\n`;
      if (effectiveCTC) {
        const claimed = effectiveCTC < 500 ? effectiveCTC*100000 : effectiveCTC;
        const pct = Math.round(((claimed-derivedAnnual)/derivedAnnual)*1000)/10;
        validatorCtx += `PF-CTC Check: Claimed=₹${(claimed/100000).toFixed(2)}L vs PF-Derived=₹${(derivedAnnual/100000).toFixed(2)}L → ${pct>0?'+':''}${pct}%\n`;
      }
    }
  }

  // Bank validators
  const bankDoc = documents.find(d => d.detectedType === "bank");
  if (bankDoc) {
    const credits = parseBankCredits(bankDoc.text || "");
    if (credits.length > 0) {
      const avgCredit = credits.reduce((a,b)=>a+b,0) / credits.length;
      const derivedAnnual = Math.round(avgCredit * 12 * 1.15);
      validatorCtx += `\nBANK VALIDATORS:\nCredits: ${credits.slice(0,6).map(c=>'₹'+c.toLocaleString('en-IN')).join(', ')}${credits.length>6?'...':''}\nAvg monthly=₹${Math.round(avgCredit).toLocaleString('en-IN')} → Derived Annual≈₹${(derivedAnnual/100000).toFixed(2)}L\n`;
      if (effectiveCTC) {
        const claimed = effectiveCTC < 500 ? effectiveCTC*100000 : effectiveCTC;
        const pct = Math.round(((claimed-derivedAnnual)/derivedAnnual)*1000)/10;
        validatorCtx += `Bank-CTC Check: Claimed=₹${(claimed/100000).toFixed(2)}L vs Bank-Derived=₹${(derivedAnnual/100000).toFixed(2)}L → ${pct>0?'+':''}${pct}%\n`;
      }
    }
  }

  // Build doc sections with per-type char limits applied
  const docSections = documents.map((d, i) => {
    const label = (d.detectedType||"unknown").toUpperCase();
    if (d.base64Image) return `DOC ${i+1} [${label}] (${d.fileName}): [IMAGE — processed by vision model]`;
    return `--- DOC ${i+1}: ${d.fileName} [${label}] ---\n${(d.text||"")}`;
  }).join("\n\n");

  const schema = SCHEMAS[analysisMode] || SCHEMAS.unknown_docs;

  const modeSpecificRules = {
    resume_only: `
RESUME ANALYSIS RULES (gap_analysis mode):
- Extract ALL companies with from/to dates in YYYY-MM format.
- Gap: any period ≥ 1 month between consecutive jobs.
- Severity: SHORT=1-2mo, MEDIUM=3-5mo, LONG=≥6mo.
- Overall verdict: CLEAN if total_gap_months<3, else GAPS_FOUND.
`,
    epfo_crosscheck: `
EPFO CROSS-CHECK RULES:
- Match resume companies against EPFO (fuzzy match, abbreviations ok).
- Flag companies only in resume as suspicious.
- inflation_years = resume_total - epfo_total.
`,
    ctc_verification: `
CTC VERIFICATION RULES:
- claimed_ctc is already in rupees.
- salary_slip: check math integrity (gross - deductions = net; PF=12% of basic, cap ₹1800).
- pf_analysis: derive basic=PF/0.12, monthly≈basic/0.4, annual=monthly*12.
- bank_analysis: avg credits*12*1.15=derived annual CTC.
- inflation: NORMAL≤20%, INFLATED 20-40%, HIGHLY_INFLATED>40%.
`,
    offer_letter: `
OFFER LETTER RULES:
- Use pre-computed CIN and GST validator results exactly.
- Pre-2017 offer → GST absence is NORMAL.
- MNC → CIN absence may be NORMAL.
- Score 0-100; verdict: AUTHENTIC≥75, SUSPICIOUS 50-74, FAKE<50.
`,
    jd_resume: `
JD-RESUME MATCH RULES:
- Compute actual_years per skill from ALL company entries.
- Sum durations across roles where skill was used.
- required_years: from JD. matched: true if actual_years>=required_years.
`,
    full_analysis: `
FULL UNIFIED ANALYSIS RULES:
- Produce ALL applicable sub-analyses in one JSON response.
- skill_match: match resume skills to JD (if JD present).
- gap_analysis: employment gaps in resume.
- ctc_check: if salary/bank/epfo present, verify CTC.
- epfo_check: if EPFO present, cross-check companies.
- offer_check: if offer letter present, verify authenticity.
- fraud_risk: aggregate fraud risk from all signals.
- final_decision: single HIRE/MAYBE/REJECT with reasoning.
`,
  };

  const system = `You are ParamkaraCorp-HR, expert Indian HR fraud detection and recruitment AI.
Analysis mode: ${analysisMode.toUpperCase()}
${validatorCtx}
STRICT RULES:
1. Output ONLY valid JSON. Zero prose, zero markdown outside JSON.
2. verdict_reason, recommendation_reason: ONE short sentence (max 20 words).
3. strengths/gaps arrays: max 4 items, max 8 words each.
4. All monetary values in rupees (₹).
5. Use pre-computed validator values exactly as given.
6. Indian PF ceiling: ₹1800/month (basic cap ₹15000). GST since July 2017.
7. Scores 0-100: 80+=SHORTLIST, 55-79=MAYBE, <55=REJECT.
${modeSpecificRules[analysisMode] || ''}
HOW TO COMPUTE actual_years PER SKILL:
- Read ALL company entries: name, start date, end date (or "Present").
- For each role, identify skills used based on job title and description.
- Compute duration: (end_year-start_year)+(end_month-start_month)/12. Round to 1 decimal.
- Sum durations across roles where that skill was used.

REQUIRED JSON SCHEMA:
${schema}`;

  return { system, userPrompt: docSections, analysisMode, effectiveCTC };
}

// ── Schemas ────────────────────────────────────────────────────────────────────
const SCHEMAS = {

  // NEW: Unified full analysis schema
  full_analysis: `{
  "type": "full_analysis",
  "candidate": {"name": "string|null", "total_experience_years": number, "current_role": "string|null"},
  "skill_match": {
    "present": boolean,
    "match_percentage": number|null,
    "mandatory_skills_met": number|null,
    "mandatory_skills_total": number|null,
    "recommendation": "SHORTLIST"|"MAYBE"|"REJECT"|null,
    "recommendation_reason": "string|null",
    "skills": [{"skill": "string", "required_years": number|null, "actual_years": number, "matched": boolean}]
  },
  "gap_analysis": {
    "present": boolean,
    "companies": [{"company": "string", "from": "YYYY-MM", "to": "YYYY-MM|Present", "role": "string"}],
    "gaps": [{"from_company": "string", "to_company": "string", "gap_months": number, "severity": "SHORT"|"MEDIUM"|"LONG"}],
    "total_gap_months": number,
    "verdict": "CLEAN"|"GAPS_FOUND"
  },
  "ctc_check": {
    "present": boolean,
    "claimed_ctc_lpa": number|null,
    "overall_verdict": "GENUINE"|"SUSPICIOUS"|"HIGHLY_INFLATED"|"INSUFFICIENT_DATA"|null,
    "confidence": "HIGH"|"MEDIUM"|"LOW"|null,
    "sources_used": ["salary_slip"|"pf_history"|"bank_statement"],
    "salary_slip": {"present": boolean, "verdict": "GENUINE"|"SUSPICIOUS"|"FAKE"|null, "derived_annual_ctc": number|null, "inflation_percent": number|null, "inflation_verdict": "NORMAL"|"INFLATED"|"HIGHLY_INFLATED"|"NO_DATA"|null, "red_flags": ["string"]},
    "pf_analysis": {"present": boolean, "derived_annual_ctc": number|null, "inflation_percent": number|null, "inflation_verdict": "NORMAL"|"INFLATED"|"HIGHLY_INFLATED"|"NO_DATA"|null},
    "bank_analysis": {"present": boolean, "derived_annual_ctc": number|null, "inflation_percent": number|null, "inflation_verdict": "NORMAL"|"INFLATED"|"HIGHLY_INFLATED"|"NO_DATA"|null}
  },
  "epfo_check": {
    "present": boolean,
    "verdict": "MATCH"|"PARTIAL"|"MISMATCH"|null,
    "verdict_reason": "string|null",
    "inflation_years": number|null,
    "red_flags": ["string"]
  },
  "offer_check": {
    "present": boolean,
    "verdict": "AUTHENTIC"|"SUSPICIOUS"|"FAKE"|null,
    "overall_score": number|null,
    "verdict_reason": "string|null"
  },
  "fraud_risk": "LOW"|"MEDIUM"|"HIGH",
  "fraud_signals": ["string"],
  "positive_signals": ["string"],
  "final_decision": "HIRE"|"MAYBE"|"REJECT",
  "final_decision_reason": "string"
}`,

  resume_only: `{
  "type": "employment",
  "mode": "gap_analysis",
  "candidate_name": "string|null",
  "companies": [{"company": "string", "from": "YYYY-MM", "to": "YYYY-MM|Present", "role": "string"}],
  "gaps": [{"from_company": "string", "to_company": "string", "gap_months": number, "severity": "SHORT"|"MEDIUM"|"LONG", "label": "string"}],
  "total_gap_months": number,
  "total_experience_years": number,
  "verdict": "CLEAN"|"GAPS_FOUND",
  "verdict_reason": "string",
  "red_flags": ["string"]
}`,

  jd_resume: `{
  "type": "jd_resume",
  "candidate": {"name": "string", "total_experience_years": number, "current_role": "string", "current_company": "string"},
  "skill_match": [{"skill": "string", "required_years": number|null, "actual_years": number, "matched": boolean, "used_at": "company1, company2"}],
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

  multi_rank: `{
  "type": "multi_rank",
  "role": "string",
  "jd_skills": ["skill1", "skill2"],
  "candidates": [
    {"name": "string", "total_experience_years": number, "current_role": "string",
     "match_percentage": number, "mandatory_met": number, "mandatory_total": number,
     "recommendation": "SHORTLIST"|"MAYBE"|"REJECT", "recommendation_reason": "string",
     "skill_match": [{"skill": "string", "required_years": number|null, "actual_years": number, "matched": boolean}]}
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

  ctc_verification: `{
  "type": "ctc_verification",
  "claimed_ctc": number,
  "claimed_ctc_lpa": number,
  "overall_verdict": "GENUINE"|"SUSPICIOUS"|"HIGHLY_INFLATED"|"INSUFFICIENT_DATA",
  "overall_verdict_reason": "string",
  "confidence": "HIGH"|"MEDIUM"|"LOW",
  "sources_used": ["salary_slip"|"pf_history"|"bank_statement"],
  "salary_slip": {
    "present": boolean,
    "verdict": "GENUINE"|"SUSPICIOUS"|"FAKE"|null,
    "overall_score": number|null,
    "extracted": {"employee_name": "string|null", "company_name": "string|null", "slip_month": "string|null", "basic": number|null, "gross": number|null, "net": number|null, "pf_employee": number|null},
    "rule_checks": [{"name": "string", "status": "pass"|"fail"|"neutral", "finding": "string"}],
    "derived_annual_ctc": number|null,
    "inflation_percent": number|null,
    "inflation_verdict": "NORMAL"|"INFLATED"|"HIGHLY_INFLATED"|"NO_DATA"|null,
    "red_flags": ["string"]
  },
  "pf_analysis": {
    "present": boolean, "pf_amounts": [number], "avg_pf_monthly": number|null,
    "derived_basic": number|null, "derived_annual_ctc": number|null,
    "inflation_percent": number|null, "inflation_verdict": "NORMAL"|"INFLATED"|"HIGHLY_INFLATED"|"NO_DATA"|null,
    "consistency": "CONSISTENT"|"INCONSISTENT"|"INSUFFICIENT_DATA", "note": "string"
  },
  "bank_analysis": {
    "present": boolean, "credits_found": number, "avg_monthly_credit": number|null,
    "derived_annual_ctc": number|null, "inflation_percent": number|null,
    "inflation_verdict": "NORMAL"|"INFLATED"|"HIGHLY_INFLATED"|"NO_DATA"|null,
    "consistency": "CONSISTENT"|"INCONSISTENT"|"INSUFFICIENT_DATA", "note": "string"
  },
  "red_flags": ["string"],
  "positive_signals": ["string"]
}`,

  epfo_crosscheck: `{
  "type": "employment",
  "mode": "epfo_crosscheck",
  "candidate_name": "string|null",
  "matched": [{"resume_company": "string", "epfo_company": "string", "date_match": boolean, "experience_gap_months": number}],
  "only_in_resume": [{"company": "string", "flag": "string"}],
  "only_in_epfo": [{"company": "string"}],
  "experience_summary": {"resume_total_years": number, "epfo_total_years": number, "inflation_years": number},
  "verdict": "MATCH"|"PARTIAL"|"MISMATCH",
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
     "verdict": "CLEAN"|"GAPS_FOUND", "total_gap_months": number}
  ]
}`,

  unknown_docs: `{
  "type": "unknown",
  "detected_content": "string",
  "findings": ["string"],
  "verdict": "string",
  "red_flags": ["string"]
}`,
};

// ── Conversational fallback ──────────────────────────────────────────────────────
async function chat(message, history) {
  const msgs = [
    { role: "system", content: `You are ParamkaraCorp-HR Assistant for Indian HR professionals. Help with: resume screening, JD matching, offer letter fraud, salary verification, EPFO checks. Be concise — max 3 sentences.` },
    ...(history||[]).slice(-4),
    { role: "user", content: message },
  ];
  return { type: "conversation", reply: await groq(msgs, false, 300) };
}

// ── Main handler ────────────────────────────────────────────────────────────────
exports.handler = async (event) => {
  if (event.httpMethod === "OPTIONS") return { statusCode: 200, headers: CORS, body: "" };
  if (event.httpMethod !== "POST")    return { statusCode: 405, headers: CORS, body: JSON.stringify({ error: "Method not allowed" }) };
  if (!GROQ_API_KEY)                  return { statusCode: 500, headers: { ...CORS, "Content-Type": "application/json" }, body: JSON.stringify({ error: "GROQ_API_KEY not set" }) };

  try {
    const body = JSON.parse(event.body || "{}");
    let { message, documents, history, claimedCTC } = body;

    // ── Pure conversation ──────────────────────────────────────────────────────
    if (!documents || documents.length === 0) {
      const r = await chat(message || "Hello", history);
      return { statusCode: 200, headers: { ...CORS, "Content-Type": "application/json" }, body: JSON.stringify(r) };
    }

    // ── Server-side type detection ─────────────────────────────────────────────
    for (const doc of documents) {
      if (!doc.detectedType || doc.detectedType === "unknown")
        doc.detectedType = detectType(doc.text || "", doc.fileName || "");
    }

    // ── GATE 1: Resume MUST be present ────────────────────────────────────────
    const types = documents.map(d => d.detectedType);
    const hasResume = types.includes("resume");
    if (!hasResume) {
      return {
        statusCode: 400,
        headers: { ...CORS, "Content-Type": "application/json" },
        body: JSON.stringify({
          type: "error",
          code: "RESUME_REQUIRED",
          error: "Resume is mandatory. Please upload a resume before running analysis.",
        }),
      };
    }

    // ── GATE 2: CTC needed check ───────────────────────────────────────────────
    const needsCTCDocs = types.some(t => ["salary","bank","epfo"].includes(t));
    const hasCTCValue  = !!claimedCTC || documents.some(d => !!extractCTC(d.text||""));
    if (needsCTCDocs && !hasCTCValue) {
      return {
        statusCode: 200,
        headers: { ...CORS, "Content-Type": "application/json" },
        body: JSON.stringify({
          type: "needs_ctc",
          detected_docs: documents.map(d => ({ fileName: d.fileName, detectedType: d.detectedType })),
        }),
      };
    }

    // ── PIPELINE: Apply doc limits (priority sort + MAX_DOCS cap + truncation) ─
    const limitedDocs = applyDocLimits(documents);
    const droppedCount = documents.length - limitedDocs.length;

    // ── Build prompt & call LLM ────────────────────────────────────────────────
    const { system, userPrompt, analysisMode, effectiveCTC } = buildPrompt(limitedDocs, claimedCTC);
    const hasImage = limitedDocs.some(d => d.base64Image);

    let result;

    if (hasImage) {
      const blocks = [];
      for (const doc of limitedDocs) {
        if (doc.base64Image) {
          blocks.push({ type: "image_url", image_url: { url: `data:${doc.imageMediaType};base64,${doc.base64Image}` } });
          blocks.push({ type: "text", text: `Image: ${doc.fileName}` });
        } else if (doc.text) {
          blocks.push({ type: "text", text: `--- ${doc.fileName} [${doc.detectedType}] ---\n${doc.text}` });
        }
      }
      blocks.push({ type: "text", text: `\nReturn JSON per schema:\n${system.split("REQUIRED JSON SCHEMA:")[1]||""}` });

      const vRes = await fetch("https://api.groq.com/openai/v1/chat/completions", {
        method: "POST",
        headers: { Authorization: `Bearer ${GROQ_API_KEY}`, "Content-Type": "application/json" },
        body: JSON.stringify({
          model: VISION_MODEL,
          messages: [{ role: "system", content: system }, { role: "user", content: blocks }],
          temperature: 0.05, max_tokens: 3000,
          response_format: { type: "json_object" }
        }),
      });
      if (!vRes.ok) {
        // Fall back to text model with image placeholder
        result = await groqWithRetry([
          { role: "system", content: system },
          { role: "user", content: "[IMAGE — analyze based on filename and other docs]\n" + userPrompt }
        ], true, 3000);
      } else {
        const vd = await vRes.json();
        const ct = vd.choices?.[0]?.message?.content ?? "{}";
        const m  = ct.match(/\{[\s\S]*\}/);
        result = m ? JSON.parse(m[0]) : JSON.parse(ct);
      }
    } else {
      const isMulti   = analysisMode === "multi_rank";
      const isFull    = analysisMode === "full_analysis";
      const tokenLimit = isMulti ? 3500 : isFull ? 3500 : analysisMode === "ctc_verification" ? 3000 : 2500;
      result = await groqWithRetry(
        [{ role: "system", content: system }, { role: "user", content: userPrompt }],
        true, tokenLimit
      );
    }

    if (!result.type) result.type = analysisMode;

    // ── Attach server-computed CTC inflation for salary docs ───────────────────
    if (effectiveCTC) {
      const salDoc = limitedDocs.find(d => d.detectedType === "salary");
      if (salDoc) {
        const gross = extractAmt(salDoc.text,"Gross Earnings","Total Earnings","Gross Salary","Gross");
        const basic = extractAmt(salDoc.text,"Basic Salary","Basic Pay","Basic");
        if (gross) {
          const empPF   = basic ? Math.min(basic*0.12,1800) : 1800;
          const grat    = basic ? basic*0.0481 : 0;
          const derived = Math.round((gross+empPF+grat)*12);
          const claimed = effectiveCTC < 500 ? effectiveCTC*100000 : effectiveCTC;
          const pct     = Math.round(((claimed-derived)/derived)*1000)/10;
          const obj     = {
            derived_annual_ctc: derived, claimed_ctc_annual: claimed, inflation_percent: pct,
            verdict: Math.abs(pct)<=15?"NORMAL":pct>30?"HIGHLY_INFLATED":"INFLATED"
          };
          if (result.salary_check)                      result.salary_check.ctc_inflation = obj;
          else if (result.ctc_check?.salary_slip)       result.ctc_check.salary_slip.server_ctc = obj;
          else                                          result.ctc_inflation = obj;
        }
      }
    }

    // ── Attach metadata ────────────────────────────────────────────────────────
    result._meta = {
      docs_analyzed: limitedDocs.length,
      docs_dropped: droppedCount,
      analysis_mode: analysisMode,
    };

    return {
      statusCode: 200,
      headers: { ...CORS, "Content-Type": "application/json" },
      body: JSON.stringify(result)
    };

  } catch (err) {
    console.error("Fatal:", err.stack || err);
    return {
      statusCode: 500,
      headers: { ...CORS, "Content-Type": "application/json" },
      body: JSON.stringify({ error: err.message || "Internal server error", code: "INTERNAL_ERROR" })
    };
  }
};
