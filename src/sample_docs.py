"""
Life Insurance sample documents for demo — v2
Covers: Term Policy, ULIP, Health Rider, ACORD COI, Premium Receipt
"""

SAMPLE_DOCS = {
    "life_term_policy": """
# LIFE INSURANCE POLICY DOCUMENT
Policy Number: LIP-2024-TERM-00892
Policy Type: Pure Term Insurance Plan
Insured (Life Assured): Rajesh Kumar Sharma
Date of Birth: 15 March 1985
Policy Term: 30 Years
Premium Payment Term: 30 Years
Sum Assured: INR 1,00,00,000 (One Crore)
Annual Premium: INR 12,450
Policy Commencement Date: 01 January 2024
Policy Maturity Date: 31 December 2053

## SECTION 1: DEFINITIONS

1.1 Sum Assured means the guaranteed amount payable on death of the Life Assured during the policy term, being INR 1,00,00,000 (Rupees One Crore only).

1.2 Nominee means the person nominated by the Policyholder to receive the death benefit. Current Nominee: Priya Sharma (Spouse), Relationship: Wife.

1.3 Premium means the amount payable by the Policyholder to keep this policy in force. Annual Premium: INR 12,450 payable on 1st January each year.

1.4 Grace Period means a period of 30 days from the premium due date within which the premium may be paid without the policy lapsing.

1.5 Revival Period means the period of 5 years from the date of first unpaid premium during which the lapsed policy may be revived.

## SECTION 2: BENEFITS

2.1 Death Benefit
On death of the Life Assured during the Policy Term, the Sum Assured of INR 1,00,00,000 shall be payable to the Nominee, provided the policy is in force on the date of death.

2.2 Survival Benefit
This is a Pure Term Plan. No survival benefit or maturity benefit is payable if the Life Assured survives the full policy term.

2.3 Accidental Death Benefit Rider (Optional)
If the Accidental Death Benefit Rider is attached, an additional INR 50,00,000 shall be payable on accidental death, over and above the base Sum Assured.

## SECTION 3: EXCLUSIONS

3.1 The death benefit shall NOT be payable in the following circumstances:
(a) Suicide within 12 months of policy commencement or revival
(b) Death due to participation in adventure sports or hazardous activities not declared at proposal
(c) Death due to war, terrorism, or civil commotion
(d) Death under the influence of alcohol or controlled substances
(e) Pre-existing conditions not disclosed at the time of proposal

3.2 Non-Disclosure Exclusion
If the Policyholder has made any material misrepresentation or non-disclosure of material facts in the Proposal Form, the Company reserves the right to repudiate the claim.

## SECTION 4: PREMIUM CONDITIONS

| Premium Mode | Amount (INR) | Loading |
|---|---|---|
| Annual | 12,450 | Nil |
| Semi-Annual | 6,472 | 3.5% |
| Quarterly | 3,300 | 5.9% |
| Monthly | 1,120 | 7.2% |

4.1 Medical Expenses for Organ Donor
Medical expenses incurred for the extraction of the required organ from the organ donor are covered under this policy subject to the overall Sum Assured limit. This benefit applies when organ transplant is necessitated due to a covered critical illness.

4.2 Premium Waiver on Critical Illness
All future premiums are waived if the Life Assured is diagnosed with any of the 36 listed critical illnesses, and the policy continues in full force.

## SECTION 5: LAPSE AND REVIVAL

5.1 Lapse
If the premium is not paid within the grace period of 30 days, the policy lapses and all benefits cease.

5.2 Revival
A lapsed policy may be revived within 5 years of first unpaid premium by:
(a) Payment of all arrear premiums with interest at 9% per annum
(b) Submission of satisfactory evidence of good health
(c) Payment of revival charges as applicable

## SECTION 6: FREE LOOK PERIOD

The Policyholder may return this policy within 30 days of receipt (Free Look Period) if not satisfied with the terms. The Company shall refund the premium paid after deducting stamp duty and medical examination charges.
""",

    "ulip_policy": """
# UNIT LINKED INSURANCE PLAN (ULIP)
Policy Number: ULIP-2024-MH-05521
Product Name: Wealth Plus Growth Plan
Life Assured: Ananya Mehta
Sum Assured: INR 50,00,000
Annual Premium: INR 1,00,000
Policy Term: 20 Years
Lock-in Period: 5 Years

## FUND OPTIONS

| Fund Name | Risk Profile | Current NAV (per unit) | Asset Allocation |
|---|---|---|---|
| Equity Growth Fund | High | INR 45.23 | 80% Equity, 20% Debt |
| Balanced Advantage Fund | Medium | INR 28.67 | 50% Equity, 50% Debt |
| Secure Bond Fund | Low | INR 18.12 | 100% Debt |
| Liquid Fund | Very Low | INR 12.05 | Money Market |

## SECTION 1: UNIT ALLOCATION

1.1 Premium Allocation Charges

| Policy Year | Allocation Charge |
|---|---|
| Year 1 | 5% of premium |
| Year 2–5 | 4% of premium |
| Year 6 onwards | Nil |

1.2 Fund Management Charges: 1.35% per annum on fund value.

1.3 Mortality Charges: Deducted monthly based on Sum at Risk (Sum Assured minus Fund Value).

## SECTION 2: BENEFITS

2.1 Death Benefit
Higher of:
(a) Sum Assured = INR 50,00,000, OR
(b) Fund Value at date of death
Whichever is higher shall be paid to the Nominee.

2.2 Maturity Benefit
On survival to maturity, the Fund Value (NAV × number of units held) is paid.

2.3 Partial Withdrawal
Permitted after the 5-year lock-in period. Maximum 25% of Fund Value per year. Minimum balance of INR 5,000 must be maintained.

## SECTION 3: SURRENDER VALUE

| Year of Surrender | Discontinuance Charge |
|---|---|
| Year 1 | INR 6,000 or 6% of AP (lower) |
| Year 2 | INR 5,000 or 4% of AP (lower) |
| Year 3 | INR 4,000 or 3% of AP (lower) |
| Year 4 | INR 2,000 or 2% of AP (lower) |
| Year 5+ | Nil |

3.1 Surrender before lock-in: Fund value after discontinuance charges moved to Discontinued Policy Fund earning 4% p.a. Paid at end of lock-in.

## SECTION 4: MEDICAL EXPENSES

4.1 Critical Illness Benefit
On diagnosis of 36 listed critical illnesses, INR 25,00,000 (50% of Sum Assured) shall be paid as a lump sum. Policy continues for remaining Sum Assured.

4.2 Hospital Cash Benefit Rider
INR 2,000 per day of hospitalisation (up to 60 days per year) if rider is opted. Medical expenses for organ extraction for organ donation surgery are covered up to INR 5,00,000 under the Health Plus rider.
""",

    "health_rider": """
# HEALTH PLUS RIDER — CERTIFICATE
Rider Number: HPR-2024-00445
Base Policy Number: LIP-2024-TERM-00892
Rider Type: Critical Illness + Surgical Benefit Rider
Sum Assured (Rider): INR 10,00,000
Rider Premium: INR 2,850 per annum
Rider Term: 20 Years

## COVERED CONDITIONS

| Critical Illness | Benefit (% of Rider SA) | Waiting Period |
|---|---|---|
| Cancer (Life Threatening) | 100% | 90 days |
| First Heart Attack | 100% | 90 days |
| Stroke | 100% | 90 days |
| Kidney Failure | 100% | 90 days |
| Major Organ Transplant | 100% | 90 days |
| Coronary Artery Disease | 75% | 90 days |
| Multiple Sclerosis | 75% | 90 days |
| Total Permanent Disability | 100% | Nil |

## SECTION 1: ORGAN DONOR BENEFIT

1.1 Medical expenses incurred for the extraction of the required organ from the organ donor, necessitated by a covered organ transplant, shall be reimbursed up to INR 2,50,000 under this rider.

1.2 The organ donor must be a living donor or deceased donor as certified by a registered hospital.

1.3 Organ types covered: Kidney, Liver (partial), Heart, Lung, Pancreas, Cornea.

1.4 Expenses covered include: pre-operative investigations, surgical fees, hospitalisation, post-operative care up to 30 days.

## SECTION 2: EXCLUSIONS

2.1 This rider does not cover:
(a) Cosmetic or aesthetic surgery
(b) Dental treatment or surgery
(c) Pregnancy-related expenses
(d) Mental illness or psychiatric conditions
(e) Self-inflicted injuries
(f) Experimental treatments not approved by IRDA

## SECTION 3: CLAIM PROCEDURE

3.1 Claim Intimation: Within 30 days of diagnosis or surgery.
3.2 Documents Required:
- Duly filled claim form
- Hospital discharge summary
- All investigation reports
- Treating doctor's certificate
- KYC documents of claimant
""",

    "acord_certificate": """
# ACORD 25 — CERTIFICATE OF LIABILITY INSURANCE
Certificate Number: COI-2024-MH-00123
Date Issued: 15 January 2024

## INSURED INFORMATION
Insured Name: Hamilton Logistics Pvt Ltd
Address: 501 Business Park, Andheri East, Mumbai 400069
Phone: +91 22 4567 8900

## INSURER INFORMATION

| Insurer | NAIC # | Policy Type | Policy Number | Effective | Expiry | Limit |
|---|---|---|---|---|---|---|
| National Insurance Co | IN-NIL | Commercial General Liability | CGL-2024-MH-892 | 01/01/2024 | 31/12/2024 | 1,00,00,000 |
| United India Insurance | IN-UIL | Workers Compensation | WC-2024-MH-445 | 01/01/2024 | 31/12/2024 | Statutory |
| Oriental Insurance | IN-OIL | Commercial Auto | CA-2024-MH-234 | 01/01/2024 | 31/12/2024 | 50,00,000 |

## GENERAL LIABILITY COVERAGE

| Coverage Type | Each Occurrence | Aggregate |
|---|---|---|
| General Aggregate | — | INR 2,00,00,000 |
| Products-Completed Operations | INR 1,00,00,000 | INR 2,00,00,000 |
| Personal & Advertising Injury | INR 1,00,00,000 | — |
| Each Occurrence | INR 1,00,00,000 | — |
| Damage to Rented Premises | INR 10,00,000 | — |
| Medical Expenses | INR 5,00,000 | — |

## CERTIFICATE HOLDER
Hamilton Port Trust Authority
Port Trust Building, Mumbai Port, Mumbai 400001

This certificate is issued as a matter of information only and confers no rights upon the certificate holder. ACORD 25 (2016/03)
""",

    "premium_receipt": """
# PREMIUM RECEIPT
Receipt Number: PR-2024-892-JAN
Policy Number: LIP-2024-TERM-00892
Policyholder: Rajesh Kumar Sharma
Date of Receipt: 05 January 2024
Mode of Payment: NEFT/Online Transfer
Transaction Reference: NEFT2024010500892

## PAYMENT DETAILS

| Description | Amount (INR) |
|---|---|
| Basic Premium | 12,450.00 |
| Health Plus Rider Premium | 2,850.00 |
| Accidental Death Rider Premium | 1,200.00 |
| Goods and Services Tax (GST) @ 18% | 2,970.00 |
| Total Premium Received | 19,470.00 |

Payment Status: CONFIRMED
Payment Date: 05 January 2024
Next Premium Due: 01 January 2025
Policy Status: IN FORCE
Premium Period: Annual (01 Jan 2024 – 31 Dec 2024)

This receipt is system generated and does not require a signature.
"""
}


def get_sample_queries():
    return [
        {
            "query": "Is organ donor medical expenses covered under the policy?",
            "relevant_sections": ["SECTION 4: PREMIUM CONDITIONS", "SECTION 1: ORGAN DONOR BENEFIT"],
        },
        {
            "query": "What is the sum assured for the term policy?",
            "relevant_sections": ["SECTION 1: DEFINITIONS", "SECTION 2: BENEFITS"],
        },
        {
            "query": "What happens if premium is not paid within grace period?",
            "relevant_sections": ["SECTION 5: LAPSE AND REVIVAL"],
        },
        {
            "query": "What are the fund options and their NAV in the ULIP?",
            "relevant_sections": ["FUND OPTIONS"],
        },
        {
            "query": "What is the discontinuance charge if I surrender in Year 2?",
            "relevant_sections": ["SECTION 3: SURRENDER VALUE"],
        },
        {
            "query": "What documents are needed to file a critical illness claim?",
            "relevant_sections": ["SECTION 3: CLAIM PROCEDURE"],
        },
        {
            "query": "What is the total premium paid in January 2024 receipt?",
            "relevant_sections": ["PAYMENT DETAILS"],
        },
        {
            "query": "What is the general aggregate limit in the ACORD certificate?",
            "relevant_sections": ["GENERAL LIABILITY COVERAGE"],
        },
    ]
