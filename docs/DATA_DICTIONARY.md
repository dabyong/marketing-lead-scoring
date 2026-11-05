# Data Dictionary — Bank Marketing Lead Ranking

## Source and target

UCI `bank-additional-full.csv`: 41,188 observations / 21 columns. This historical contact dataset has no unique customer IDs or precise timestamps. CSV delimiter is `;`. Original target `y` is binary term-deposit subscription (`yes` or `no`). `target` in processed data maps `yes` to 1 and `no` to 0.

## Seven scoring fields

| Field | Type | Meaning / handling |
|---|---|---|
| `default` | category | Credit default status: `yes`, `no`, or `unknown`. |
| `housing` | category | Housing loan: `yes`, `no`, or `unknown`. |
| `loan` | category | Personal loan: `yes`, `no`, or `unknown`. |
| `poutcome` | category | Outcome of previous marketing campaign: `success`, `failure`, `nonexistent`. |
| `previous` | nonnegative integer | Number of previous campaign contacts. |
| `prior_contacted` | 0 / 1 | Derived as `pdays != 999`. |
| `days_since_previous` | nonnegative integer or blank | Derived from `pdays`; blank when `pdays = 999`. The original 999 is a sentinel, **not a duration**. |

For synthetic demo uploads, **include exactly these seven columns**, no ID, `y`, outcomes, `duration`, `age`, or extra columns. Do not upload actual personal/customer records. The uploader is a schema validator; it cannot verify synthetic provenance.

## Original non-scoring fields

| Column | Treatment / reason |
|---|---|
| `age` | Excluded from model; fairness audit only. |
| `marital` | Excluded from model; fairness audit only. |
| `job` | Removed during fairness-sensitive ablation; used for audit. |
| `education` | Removed during fairness-sensitive ablation; used for audit. |
| `contact` | Current-contact channel not assumed available at campaign planning. |
| `month`, `day_of_week` | Contact-time descriptors; no precise event timestamp provided. |
| `duration` | **Excluded**: length of current call, unknown before contact. |
| `campaign` | Current-campaign contact count; availability uncertain at pre-campaign time. |
| `pdays` | Source of two engineered features; sentinel 999 handled separately. |
| `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` | Historical economic context; not assumed reliably available at the planned scoring snapshot. |
| `y` | Binary outcome target, never a model feature. |

`source_row_id` is an internal row-position marker used to verify splits. It is **never** a model feature. Exact duplicate rows were preserved for lack of verifiable unique contact IDs. Categorical `unknown` values are kept, not silently dropped.

## Dataset limitations

This data describes historical contacts, not a live CRM feed. Feature values at scoring time are assumed for research; real-world availability and legal/permitted uses would require a separate assessment. Chronological splits use original file row order rather than exact dates.
