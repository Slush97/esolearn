//! Sales Pipeline Analysis & ICP Report
//!
//! Reads cleaned pipeline data, generates charts and ML models,
//! outputs SVGs and an HTML dashboard.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use esoc_chart::interop::{ClassificationReportExt, ConfusionMatrixExt, RocCurveExt};
use esoc_chart::v2::{self, bar, grouped_bar, pie, scatter, stacked_bar, NewTheme};

use scry_learn::dataset::Dataset;
use scry_learn::explain::permutation_importance;
use scry_learn::linear::LogisticRegression;
use scry_learn::metrics::{accuracy, classification_report, confusion_matrix, roc_curve};
use scry_learn::preprocess::{StandardScaler, Transformer};
use scry_learn::split::{cross_val_score_stratified, train_test_split, ScoringFn};
use scry_learn::tree::{
    DecisionTreeClassifier, GradientBoostingClassifier, RandomForestClassifier,
};

fn home_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/home/esoc".to_string()))
}

fn figures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("report")
        .join("figures")
}

/// Convert column-major Dataset features to row-major for predict().
fn to_row_major(data: &Dataset) -> Vec<Vec<f64>> {
    let n = data.n_samples();
    let m = data.n_features();
    (0..n)
        .map(|i| (0..m).map(|j| data.features[j][i]).collect())
        .collect()
}

// ── Phase 1: Descriptive charts from the clean (string) CSV ────────────

struct CleanRecord {
    date_found: String,
    conf_start_date: String,
    status: String,
    city_tier: String,
    source: String,
    found_by: String,
    conf_month: String,
    emails_sent: f64,
    emails_received: f64,
    outreach_count: f64,
    bid_status: String,
}

/// Extract 4-digit year from an ISO date string.
fn year_of(date_str: &str) -> Option<u32> {
    if date_str.len() >= 4 {
        date_str[..4].parse().ok().filter(|&y: &u32| y >= 2015 && y <= 2026)
    } else {
        None
    }
}

fn load_clean_csv() -> Vec<CleanRecord> {
    let path = home_dir().join("pipeline_clean.csv");
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(&path)
        .expect("Failed to open pipeline_clean.csv");

    let headers: Vec<String> = rdr
        .headers()
        .unwrap()
        .iter()
        .map(|h| h.to_string())
        .collect();

    let idx = |name: &str| -> usize {
        headers
            .iter()
            .position(|h| h == name)
            .unwrap_or_else(|| panic!("Column '{name}' not found"))
    };

    let i_date_found = idx("date_found");
    let i_conf_start = idx("conf_start_date");
    let i_status = idx("status");
    let i_city_tier = idx("city_tier");
    let i_source = idx("source");
    let i_found_by = idx("found_by");
    let i_conf_month = idx("conf_month");
    let i_emails_sent = idx("emails_sent_count");
    let i_emails_received = idx("emails_received_count");
    let i_outreach = idx("outreach_count");
    let i_bid_status = idx("bid_status");
    let i_is_dup = idx("is_duplicate");

    let mut records = Vec::new();
    for result in rdr.records() {
        let row = result.unwrap();
        if row.get(i_is_dup).unwrap_or("") == "true" {
            continue;
        }
        records.push(CleanRecord {
            date_found: row.get(i_date_found).unwrap_or("").to_string(),
            conf_start_date: row.get(i_conf_start).unwrap_or("").to_string(),
            status: row.get(i_status).unwrap_or("").to_string(),
            city_tier: row.get(i_city_tier).unwrap_or("").to_string(),
            source: row.get(i_source).unwrap_or("").to_string(),
            found_by: row.get(i_found_by).unwrap_or("").to_string(),
            conf_month: row.get(i_conf_month).unwrap_or("").to_string(),
            emails_sent: row
                .get(i_emails_sent)
                .unwrap_or("")
                .parse()
                .unwrap_or(0.0),
            emails_received: row
                .get(i_emails_received)
                .unwrap_or("")
                .parse()
                .unwrap_or(0.0),
            outreach_count: row
                .get(i_outreach)
                .unwrap_or("")
                .parse()
                .unwrap_or(0.0),
            bid_status: row.get(i_bid_status).unwrap_or("").to_string(),
        });
    }
    records
}

/// Count occurrences of each value, return sorted by count descending.
fn count_values(values: &[&str]) -> Vec<(String, usize)> {
    let mut map: HashMap<&str, usize> = HashMap::new();
    for v in values {
        *map.entry(v).or_insert(0) += 1;
    }
    let mut pairs: Vec<(String, usize)> = map
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1));
    pairs
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let fig_dir = figures_dir();
    std::fs::create_dir_all(&fig_dir)?;
    let theme = NewTheme::light();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  SALES PIPELINE ANALYSIS & ICP REPORT");
    println!("═══════════════════════════════════════════════════════════════\n");

    // ═══════════════════════════════════════════════════════════════
    //  Phase 1: Descriptive Analysis from Clean CSV
    // ═══════════════════════════════════════════════════════════════
    let records = load_clean_csv();
    println!("Loaded {} non-duplicate records\n", records.len());

    // ── 1. Pipeline Status Funnel ────────────────────────────────
    {
        let statuses: Vec<&str> = records.iter().map(|r| r.status.as_str()).collect();
        let counts = count_values(&statuses);
        let labels: Vec<&str> = counts.iter().map(|(k, _)| k.as_str()).collect();
        let vals: Vec<f64> = counts.iter().map(|(_, v)| *v as f64).collect();

        println!("── PIPELINE STATUS ──");
        for (label, count) in &counts {
            println!("  {:>20}  {count:>5}", label);
        }
        println!();

        bar(&labels, &vals)
            .title("Pipeline Status Distribution")
            .x_label("Status")
            .y_label("Count")
            .theme(theme.clone())
            .size(900.0, 500.0)
            .build()
            .save_svg(fig_dir.join("01_status_funnel.svg").to_str().unwrap())?;
    }

    // ── 2. City Tier Breakdown ───────────────────────────────────
    {
        let tiers: Vec<&str> = records
            .iter()
            .filter(|r| !r.city_tier.is_empty())
            .map(|r| r.city_tier.as_str())
            .collect();
        let counts = count_values(&tiers);
        let labels: Vec<&str> = counts.iter().map(|(k, _)| k.as_str()).collect();
        let vals: Vec<f64> = counts.iter().map(|(_, v)| *v as f64).collect();

        println!("── CITY TIER ──");
        for (label, count) in &counts {
            println!("  {:>20}  {count:>5}", label);
        }
        println!();

        pie(&vals, &labels)
            .title("Conference City Tier Distribution")
            .theme(theme.clone())
            .size(600.0, 600.0)
            .build()
            .save_svg(fig_dir.join("02_city_tier_pie.svg").to_str().unwrap())?;
    }

    // ── 3. Lead Source Effectiveness ─────────────────────────────
    {
        let top_sources = [
            "google_search",
            "url_on_sheet",
            "linksheets",
            "cvent",
            "hotel_search",
            "linkedin",
            "10times",
            "city_type_in",
        ];

        let mut src_total: HashMap<&str, usize> = HashMap::new();
        let mut src_replied: HashMap<&str, usize> = HashMap::new();

        for r in &records {
            let src = if top_sources.contains(&r.source.as_str()) {
                r.source.as_str()
            } else if r.source.is_empty() {
                continue;
            } else {
                "other"
            };
            *src_total.entry(src).or_insert(0) += 1;
            if r.emails_received > 0.0 {
                *src_replied.entry(src).or_insert(0) += 1;
            }
        }

        // Sort by total descending
        let mut sources: Vec<&str> = src_total.keys().copied().collect();
        sources.sort_by(|a, b| src_total[b].cmp(&src_total[a]));

        println!("── SOURCE EFFECTIVENESS ──");
        println!("  {:>20}  {:>6}  {:>6}  {:>8}", "Source", "Total", "Reply", "Rate");
        let mut src_labels = Vec::new();
        let mut src_counts = Vec::new();
        let mut src_rates = Vec::new();
        for src in &sources {
            let total = src_total[src];
            let replied = src_replied.get(src).copied().unwrap_or(0);
            let rate = if total > 0 {
                100.0 * replied as f64 / total as f64
            } else {
                0.0
            };
            println!("  {:>20}  {total:>6}  {replied:>6}  {rate:>7.1}%", src);
            src_labels.push(*src);
            src_counts.push(total as f64);
            src_rates.push(rate);
        }
        println!();

        // Volume chart
        bar(&src_labels, &src_counts)
            .title("Lead Volume by Source")
            .x_label("Source")
            .y_label("Number of Leads")
            .theme(theme.clone())
            .size(900.0, 500.0)
            .build()
            .save_svg(fig_dir.join("03_source_volume.svg").to_str().unwrap())?;

        // Reply rate chart
        bar(&src_labels, &src_rates)
            .title("Reply Rate by Source (%)")
            .x_label("Source")
            .y_label("Reply Rate (%)")
            .theme(theme.clone())
            .size(900.0, 500.0)
            .build()
            .save_svg(fig_dir.join("04_source_reply_rate.svg").to_str().unwrap())?;
    }

    // ── 4. Researcher Performance ────────────────────────────────
    {
        let top_researchers = [
            "Evangeline",
            "Shem",
            "Sarah",
            "Lester",
            "Judith",
            "Shiloh",
            "Prosper",
            "Mahalie",
            "Jr",
            "Camille",
            "Karen",
            "Kyser",
            "Cid",
            "Vincent",
        ];

        let mut res_total: HashMap<&str, usize> = HashMap::new();
        let mut res_replied: HashMap<&str, usize> = HashMap::new();

        for r in &records {
            let researcher = if top_researchers.contains(&r.found_by.as_str()) {
                r.found_by.as_str()
            } else {
                continue;
            };
            *res_total.entry(researcher).or_insert(0) += 1;
            if r.emails_received > 0.0 {
                *res_replied.entry(researcher).or_insert(0) += 1;
            }
        }

        let mut researchers: Vec<&str> = res_total.keys().copied().collect();
        researchers.sort_by(|a, b| res_total[b].cmp(&res_total[a]));

        println!("── RESEARCHER PERFORMANCE ──");
        println!(
            "  {:>15}  {:>6}  {:>6}  {:>8}",
            "Researcher", "Total", "Reply", "Rate"
        );

        let mut cat_labels = Vec::new();
        let mut group_labels = Vec::new();
        let mut group_vals = Vec::new();

        for &name in &researchers {
            let total = res_total[name];
            let replied = res_replied.get(name).copied().unwrap_or(0);
            let rate = if total > 0 {
                100.0 * replied as f64 / total as f64
            } else {
                0.0
            };
            println!("  {:>15}  {total:>6}  {replied:>6}  {rate:>7.1}%", name);

            // Volume
            cat_labels.push(name.to_string());
            group_labels.push("Volume".to_string());
            group_vals.push(total as f64);

            // Reply rate (scaled for visibility)
            cat_labels.push(name.to_string());
            group_labels.push("Reply Rate (x10)".to_string());
            group_vals.push(rate * 10.0);
        }
        println!();

        grouped_bar(&cat_labels, &group_labels, &group_vals)
            .title("Researcher: Volume vs Reply Rate")
            .x_label("Researcher")
            .y_label("Count / Rate×10")
            .theme(theme.clone())
            .size(1100.0, 500.0)
            .build()
            .save_svg(fig_dir.join("05_researcher_performance.svg").to_str().unwrap())?;
    }

    // ── 5. Conference Month Seasonality ──────────────────────────
    {
        let months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ];

        let mut month_total: HashMap<&str, usize> = HashMap::new();
        let mut month_replied: HashMap<&str, usize> = HashMap::new();

        for r in &records {
            if r.conf_month.is_empty() {
                continue;
            }
            let m = r.conf_month.as_str();
            *month_total.entry(m).or_insert(0) += 1;
            if r.emails_received > 0.0 {
                *month_replied.entry(m).or_insert(0) += 1;
            }
        }

        println!("── MONTH SEASONALITY ──");
        println!(
            "  {:>12}  {:>6}  {:>6}  {:>8}",
            "Month", "Total", "Reply", "Rate"
        );

        let mut cat_labels = Vec::new();
        let mut group_labels = Vec::new();
        let mut group_vals = Vec::new();

        for m in &months {
            let total = month_total.get(m).copied().unwrap_or(0);
            let replied = month_replied.get(m).copied().unwrap_or(0);
            let rate = if total > 0 {
                100.0 * replied as f64 / total as f64
            } else {
                0.0
            };
            println!("  {:>12}  {total:>6}  {replied:>6}  {rate:>7.1}%", m);

            cat_labels.push(m.to_string());
            group_labels.push("Volume".to_string());
            group_vals.push(total as f64);

            cat_labels.push(m.to_string());
            group_labels.push("Reply Rate (x50)".to_string());
            group_vals.push(rate * 50.0);
        }
        println!();

        grouped_bar(&cat_labels, &group_labels, &group_vals)
            .title("Conference Month: Volume & Reply Rate")
            .x_label("Month")
            .y_label("Count / Rate×50")
            .theme(theme.clone())
            .size(1100.0, 500.0)
            .build()
            .save_svg(fig_dir.join("06_month_seasonality.svg").to_str().unwrap())?;
    }

    // ── 6. Outreach Diminishing Returns ──────────────────────────
    {
        let max_outreach = 30;
        let mut bucket_total = vec![0usize; max_outreach + 1];
        let mut bucket_replied = vec![0usize; max_outreach + 1];

        for r in &records {
            let oc = (r.outreach_count as usize).min(max_outreach);
            bucket_total[oc] += 1;
            if r.emails_received > 0.0 {
                bucket_replied[oc] += 1;
            }
        }

        let mut x_vals = Vec::new();
        let mut y_rates = Vec::new();

        println!("── OUTREACH DIMINISHING RETURNS ──");
        println!(
            "  {:>10}  {:>6}  {:>6}  {:>8}",
            "Outreach#", "Total", "Reply", "Rate"
        );
        for i in 0..=max_outreach {
            if bucket_total[i] < 10 {
                continue; // skip sparse buckets
            }
            let rate = 100.0 * bucket_replied[i] as f64 / bucket_total[i] as f64;
            println!(
                "  {:>10}  {:>6}  {:>6}  {rate:>7.1}%",
                i, bucket_total[i], bucket_replied[i]
            );
            x_vals.push(i as f64);
            y_rates.push(rate);
        }
        println!();

        v2::line(&x_vals, &y_rates)
            .title("Reply Rate vs Outreach Count")
            .x_label("Number of Outreach Attempts")
            .y_label("Reply Rate (%)")
            .theme(theme.clone())
            .size(900.0, 500.0)
            .build()
            .save_svg(fig_dir.join("07_outreach_diminishing_returns.svg").to_str().unwrap())?;
    }

    // ── 7. Engagement Scatter: Sent vs Received ──────────────────
    {
        let emailed: Vec<&CleanRecord> = records
            .iter()
            .filter(|r| r.emails_sent > 0.0)
            .collect();

        let x: Vec<f64> = emailed.iter().map(|r| r.emails_sent).collect();
        let y: Vec<f64> = emailed.iter().map(|r| r.emails_received).collect();
        let cats: Vec<String> = emailed
            .iter()
            .map(|r| {
                if r.emails_received > 0.0 {
                    "Got Reply".to_string()
                } else {
                    "No Reply".to_string()
                }
            })
            .collect();

        scatter(&x, &y)
            .color_by(&cats)
            .title("Emails Sent vs Received")
            .x_label("Emails Sent")
            .y_label("Emails Received")
            .theme(theme.clone())
            .size(800.0, 600.0)
            .build()
            .save_svg(fig_dir.join("08_engagement_scatter.svg").to_str().unwrap())?;
    }

    // ── 8. Bid Outcomes ──────────────────────────────────────────
    {
        let bid_records: Vec<&CleanRecord> =
            records.iter().filter(|r| !r.bid_status.is_empty()).collect();

        if !bid_records.is_empty() {
            let statuses: Vec<&str> = bid_records.iter().map(|r| r.bid_status.as_str()).collect();
            let counts = count_values(&statuses);
            let labels: Vec<&str> = counts.iter().map(|(k, _)| k.as_str()).collect();
            let vals: Vec<f64> = counts.iter().map(|(_, v)| *v as f64).collect();

            println!("── BID OUTCOMES ({} bids) ──", bid_records.len());
            for (label, count) in &counts {
                println!("  {:>15}  {count:>5}", label);
            }
            println!();

            pie(&vals, &labels)
                .title(&format!("Bid Outcomes ({} total bids)", bid_records.len()))
                .theme(theme.clone())
                .size(600.0, 600.0)
                .build()
                .save_svg(fig_dir.join("09_bid_outcomes.svg").to_str().unwrap())?;
        }
    }

    // ── 9. Year-over-Year: Leads Found ─────────────────────────
    {
        let years: Vec<u32> = (2015..=2026).collect();
        let year_labels: Vec<String> = years.iter().map(|y| y.to_string()).collect();

        // Leads found per year
        let mut found_total = vec![0usize; years.len()];
        let mut found_replied = vec![0usize; years.len()];
        let mut found_bids = vec![0usize; years.len()];

        for r in &records {
            if let Some(y) = year_of(&r.date_found) {
                if let Some(i) = years.iter().position(|&yr| yr == y) {
                    found_total[i] += 1;
                    if r.emails_received > 0.0 {
                        found_replied[i] += 1;
                    }
                    if !r.bid_status.is_empty() {
                        found_bids[i] += 1;
                    }
                }
            }
        }

        let found_rates: Vec<f64> = found_total
            .iter()
            .zip(found_replied.iter())
            .map(|(&t, &r)| if t > 0 { 100.0 * r as f64 / t as f64 } else { 0.0 })
            .collect();

        println!("── YEAR-OVER-YEAR: LEADS FOUND ──");
        println!("  {:>6}  {:>6}  {:>6}  {:>8}  {:>5}", "Year", "Found", "Reply", "Rate", "Bids");
        for (i, y) in years.iter().enumerate() {
            println!(
                "  {:>6}  {:>6}  {:>6}  {:>7.1}%  {:>5}",
                y, found_total[i], found_replied[i], found_rates[i], found_bids[i]
            );
        }
        println!();

        // Volume by year
        let found_f64: Vec<f64> = found_total.iter().map(|&v| v as f64).collect();
        let year_strs: Vec<&str> = year_labels.iter().map(|s| s.as_str()).collect();

        bar(&year_strs, &found_f64)
            .title("Leads Found per Year")
            .x_label("Year")
            .y_label("Number of Leads")
            .theme(theme.clone())
            .size(900.0, 500.0)
            .build()
            .save_svg(fig_dir.join("16_yoy_leads_volume.svg").to_str().unwrap())?;

        // Reply rate by year
        bar(&year_strs, &found_rates)
            .title("Reply Rate by Year Found (%)")
            .x_label("Year")
            .y_label("Reply Rate (%)")
            .theme(theme.clone())
            .size(900.0, 500.0)
            .build()
            .save_svg(fig_dir.join("17_yoy_reply_rate.svg").to_str().unwrap())?;

        // Bids by year
        let bids_f64: Vec<f64> = found_bids.iter().map(|&v| v as f64).collect();
        bar(&year_strs, &bids_f64)
            .title("Bids Placed by Year")
            .x_label("Year")
            .y_label("Number of Bids")
            .theme(theme.clone())
            .size(900.0, 500.0)
            .build()
            .save_svg(fig_dir.join("18_yoy_bids.svg").to_str().unwrap())?;
    }

    // ── 10. Year-over-Year: Source Mix Evolution ─────────────────
    {
        let years: Vec<u32> = (2015..=2026).collect();
        let year_labels: Vec<String> = years.iter().map(|y| y.to_string()).collect();
        let key_sources = ["google_search", "url_on_sheet", "linksheets", "cvent", "linkedin"];

        let mut cat_labels = Vec::new();
        let mut group_labels = Vec::new();
        let mut group_vals = Vec::new();

        for src in &key_sources {
            for (i, y) in years.iter().enumerate() {
                let count = records
                    .iter()
                    .filter(|r| {
                        year_of(&r.date_found) == Some(*y) && r.source.as_str() == *src
                    })
                    .count();
                cat_labels.push(year_labels[i].clone());
                group_labels.push(src.to_string());
                group_vals.push(count as f64);
            }
        }

        println!("── SOURCE MIX EVOLUTION ──");
        for src in &key_sources {
            let counts: Vec<String> = years
                .iter()
                .map(|y| {
                    let c = records
                        .iter()
                        .filter(|r| year_of(&r.date_found) == Some(*y) && r.source.as_str() == *src)
                        .count();
                    format!("{c:>5}")
                })
                .collect();
            println!("  {:>15}: {}", src, counts.join(" "));
        }
        println!("  {:>15}: {}", "years", years.iter().map(|y| format!("{y:>5}")).collect::<Vec<_>>().join(" "));
        println!();

        stacked_bar(&cat_labels, &group_labels, &group_vals)
            .title("Lead Source Mix by Year")
            .x_label("Year")
            .y_label("Number of Leads")
            .theme(theme.clone())
            .size(1100.0, 500.0)
            .build()
            .save_svg(fig_dir.join("19_yoy_source_mix.svg").to_str().unwrap())?;
    }

    // ── 11. Year-over-Year: Researcher Activity ──────────────────
    {
        let years: Vec<u32> = (2015..=2026).collect();
        let year_labels: Vec<String> = years.iter().map(|y| y.to_string()).collect();
        let key_researchers = [
            "Evangeline", "Shem", "Sarah", "Lester", "Prosper", "Shiloh",
        ];

        let mut cat_labels = Vec::new();
        let mut group_labels = Vec::new();
        let mut group_vals = Vec::new();

        println!("── RESEARCHER ACTIVITY BY YEAR ──");
        for res in &key_researchers {
            let mut counts = Vec::new();
            for (i, y) in years.iter().enumerate() {
                let count = records
                    .iter()
                    .filter(|r| {
                        year_of(&r.date_found) == Some(*y) && r.found_by.as_str() == *res
                    })
                    .count();
                counts.push(count);
                cat_labels.push(year_labels[i].clone());
                group_labels.push(res.to_string());
                group_vals.push(count as f64);
            }
            println!(
                "  {:>12}: {}",
                res,
                counts.iter().map(|c| format!("{c:>5}")).collect::<Vec<_>>().join(" ")
            );
        }
        println!("  {:>12}: {}", "years", years.iter().map(|y| format!("{y:>5}")).collect::<Vec<_>>().join(" "));
        println!();

        stacked_bar(&cat_labels, &group_labels, &group_vals)
            .title("Researcher Activity by Year")
            .x_label("Year")
            .y_label("Leads Found")
            .theme(theme.clone())
            .size(1100.0, 500.0)
            .build()
            .save_svg(fig_dir.join("20_yoy_researcher_activity.svg").to_str().unwrap())?;
    }

    // ── 12. Year-over-Year: City Tier Composition ────────────────
    {
        let years: Vec<u32> = (2015..=2026).collect();
        let year_labels: Vec<String> = years.iter().map(|y| y.to_string()).collect();
        let tiers = ["major", "minor", "international", "virtual"];

        let mut cat_labels = Vec::new();
        let mut group_labels = Vec::new();
        let mut group_vals = Vec::new();

        println!("── CITY TIER COMPOSITION BY YEAR ──");
        for tier in &tiers {
            let mut counts = Vec::new();
            for (i, y) in years.iter().enumerate() {
                let count = records
                    .iter()
                    .filter(|r| {
                        year_of(&r.date_found) == Some(*y) && r.city_tier.as_str() == *tier
                    })
                    .count();
                counts.push(count);
                cat_labels.push(year_labels[i].clone());
                group_labels.push(tier.to_string());
                group_vals.push(count as f64);
            }
            println!(
                "  {:>15}: {}",
                tier,
                counts.iter().map(|c| format!("{c:>5}")).collect::<Vec<_>>().join(" ")
            );
        }
        println!("  {:>15}: {}", "years", years.iter().map(|y| format!("{y:>5}")).collect::<Vec<_>>().join(" "));
        println!();

        stacked_bar(&cat_labels, &group_labels, &group_vals)
            .title("City Tier Composition by Year")
            .x_label("Year")
            .y_label("Number of Leads")
            .theme(theme.clone())
            .size(1100.0, 500.0)
            .build()
            .save_svg(fig_dir.join("21_yoy_city_tier.svg").to_str().unwrap())?;
    }

    // ── 13. Year-over-Year: Reply Rate by City Tier ──────────────
    {
        let years: Vec<u32> = (2016..=2024).collect(); // only years with meaningful data
        let year_labels: Vec<String> = years.iter().map(|y| y.to_string()).collect();
        let tiers = ["major", "international"];

        let mut x_vals = Vec::new();
        let mut y_vals = Vec::new();
        let mut cats = Vec::new();

        println!("── REPLY RATE BY TIER OVER TIME ──");
        for tier in &tiers {
            print!("  {:>15}:", tier);
            for y in &years {
                let total = records
                    .iter()
                    .filter(|r| year_of(&r.date_found) == Some(*y) && r.city_tier.as_str() == *tier)
                    .count();
                let replied = records
                    .iter()
                    .filter(|r| {
                        year_of(&r.date_found) == Some(*y)
                            && r.city_tier.as_str() == *tier
                            && r.emails_received > 0.0
                    })
                    .count();
                let rate = if total > 0 { 100.0 * replied as f64 / total as f64 } else { 0.0 };
                print!(" {rate:>5.1}%");
                x_vals.push(*y as f64);
                y_vals.push(rate);
                cats.push(tier.to_string());
            }
            println!();
        }
        println!("  {:>15}: {}", "years", years.iter().map(|y| format!("{y:>6}")).collect::<Vec<_>>().join(" "));
        println!();

        scatter(&x_vals, &y_vals)
            .color_by(&cats)
            .title("Reply Rate: Major vs International (by year found)")
            .x_label("Year")
            .y_label("Reply Rate (%)")
            .theme(theme.clone())
            .size(900.0, 500.0)
            .build()
            .save_svg(fig_dir.join("22_yoy_tier_reply_trend.svg").to_str().unwrap())?;
    }

    // ── 14. Conference Target Year Distribution ──────────────────
    {
        let conf_years: Vec<u32> = (2022..=2027).collect();
        let year_labels: Vec<String> = conf_years.iter().map(|y| y.to_string()).collect();
        let year_strs: Vec<&str> = year_labels.iter().map(|s| s.as_str()).collect();

        let mut totals = Vec::new();
        let mut replied = Vec::new();

        println!("── CONFERENCES BY TARGET YEAR ──");
        println!("  {:>6}  {:>6}  {:>6}  {:>8}", "Year", "Total", "Reply", "Rate");
        for y in &conf_years {
            let t = records
                .iter()
                .filter(|r| year_of(&r.conf_start_date) == Some(*y))
                .count();
            let rep = records
                .iter()
                .filter(|r| {
                    year_of(&r.conf_start_date) == Some(*y) && r.emails_received > 0.0
                })
                .count();
            let rate = if t > 0 { 100.0 * rep as f64 / t as f64 } else { 0.0 };
            println!("  {:>6}  {:>6}  {:>6}  {:>7.1}%", y, t, rep, rate);
            totals.push(t as f64);
            replied.push(rep as f64);
        }
        println!();

        // Stacked: replied vs unreplied
        let unreplied: Vec<f64> = totals.iter().zip(replied.iter()).map(|(t, r)| t - r).collect();
        let mut cat_labels = Vec::new();
        let mut group_labels = Vec::new();
        let mut group_vals = Vec::new();
        for (i, _) in conf_years.iter().enumerate() {
            cat_labels.push(year_strs[i].to_string());
            group_labels.push("Replied".to_string());
            group_vals.push(replied[i]);
            cat_labels.push(year_strs[i].to_string());
            group_labels.push("No Reply".to_string());
            group_vals.push(unreplied[i]);
        }

        stacked_bar(&cat_labels, &group_labels, &group_vals)
            .title("Conferences by Target Year (Replied vs No Reply)")
            .x_label("Conference Year")
            .y_label("Count")
            .theme(theme.clone())
            .size(900.0, 500.0)
            .build()
            .save_svg(fig_dir.join("23_conf_year_engagement.svg").to_str().unwrap())?;
    }

    // ── 15. Cumulative Pipeline Growth ───────────────────────────
    {
        let years: Vec<u32> = (2015..=2026).collect();
        let year_labels: Vec<String> = years.iter().map(|y| y.to_string()).collect();

        let mut cumulative = Vec::new();
        let mut running = 0.0;

        for y in &years {
            let count = records
                .iter()
                .filter(|r| year_of(&r.date_found) == Some(*y))
                .count();
            running += count as f64;
            cumulative.push(running);
        }

        let x_years: Vec<f64> = years.iter().map(|&y| y as f64).collect();
        v2::line(&x_years, &cumulative)
            .title("Cumulative Pipeline Growth")
            .x_label("Year")
            .y_label("Total Leads (Cumulative)")
            .theme(theme.clone())
            .size(900.0, 500.0)
            .build()
            .save_svg(fig_dir.join("24_cumulative_growth.svg").to_str().unwrap())?;
    }

    // ── 16. City Tier vs Reply Rate ──────────────────────────────
    {
        let tiers = ["major", "minor", "international", "virtual"];
        let mut tier_total: HashMap<&str, usize> = HashMap::new();
        let mut tier_replied: HashMap<&str, usize> = HashMap::new();

        for r in &records {
            if r.city_tier.is_empty() {
                continue;
            }
            let t = r.city_tier.as_str();
            *tier_total.entry(t).or_insert(0) += 1;
            if r.emails_received > 0.0 {
                *tier_replied.entry(t).or_insert(0) += 1;
            }
        }

        println!("── CITY TIER REPLY RATES ──");
        let mut labels = Vec::new();
        let mut rates = Vec::new();
        for t in &tiers {
            let total = tier_total.get(t).copied().unwrap_or(0);
            let replied = tier_replied.get(t).copied().unwrap_or(0);
            let rate = if total > 0 {
                100.0 * replied as f64 / total as f64
            } else {
                0.0
            };
            println!("  {:>15}  {total:>5} total  {replied:>5} replied  {rate:.1}%", t);
            labels.push(*t);
            rates.push(rate);
        }
        println!();

        bar(&labels, &rates)
            .title("Reply Rate by City Tier (%)")
            .x_label("City Tier")
            .y_label("Reply Rate (%)")
            .theme(theme.clone())
            .size(700.0, 500.0)
            .build()
            .save_svg(fig_dir.join("10_tier_reply_rate.svg").to_str().unwrap())?;
    }

    // ═══════════════════════════════════════════════════════════════
    //  Phase 2: ML — Predict Reply Likelihood (ICP)
    // ═══════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("  MACHINE LEARNING — REPLY PREDICTION MODEL");
    println!("═══════════════════════════════════════════════════════════════\n");

    let csv_path = home_dir().join("pipeline_numeric.csv");
    let mut data = Dataset::from_csv(csv_path.to_str().unwrap(), "got_reply")?;

    println!("Loaded numeric dataset: {} samples, {} features", data.n_samples(), data.n_features());
    println!("Feature names: {:?}\n", data.feature_names);
    data.describe();

    // Scale
    let mut scaled = data.clone();
    let mut scaler = StandardScaler::new();
    scaler.fit(&scaled)?;
    scaler.transform(&mut scaled)?;

    // Split
    let (train, test) = train_test_split(&data, 0.2, 42);
    let (train_s, test_s) = train_test_split(&scaled, 0.2, 42);

    println!(
        "\nTrain: {} samples, Test: {} samples",
        train.n_samples(),
        test.n_samples()
    );

    let pos_train = train.target.iter().filter(|&&v| v == 1.0).count();
    let pos_test = test.target.iter().filter(|&&v| v == 1.0).count();
    println!(
        "Train positives: {} ({:.1}%), Test positives: {} ({:.1}%)\n",
        pos_train,
        100.0 * pos_train as f64 / train.n_samples() as f64,
        pos_test,
        100.0 * pos_test as f64 / test.n_samples() as f64,
    );

    // ── Cross-validation ─────────────────────────────────────────
    println!("── 5-FOLD STRATIFIED CROSS-VALIDATION ──");
    let scorer: ScoringFn = accuracy;

    struct CvResult {
        name: &'static str,
        mean: f64,
        std: f64,
    }

    let mut results: Vec<CvResult> = Vec::new();

    macro_rules! run_cv {
        ($name:expr, $model:expr, $data:expr) => {{
            let start = Instant::now();
            let scores = cross_val_score_stratified(&$model, &$data, 5, scorer, 42)?;
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            let mean = scores.iter().sum::<f64>() / scores.len() as f64;
            let var = scores
                .iter()
                .map(|s| (s - mean).powi(2))
                .sum::<f64>()
                / scores.len() as f64;
            let std = var.sqrt();
            println!(
                "  {:25} {mean:.4} ± {std:.4}  ({elapsed:.1} ms)",
                $name
            );
            results.push(CvResult {
                name: $name,
                mean,
                std,
            });
        }};
    }

    run_cv!(
        "Decision Tree",
        DecisionTreeClassifier::new().max_depth(8),
        data
    );
    run_cv!(
        "Random Forest",
        RandomForestClassifier::new()
            .n_estimators(100)
            .max_depth(10)
            .seed(42),
        data
    );
    run_cv!(
        "Gradient Boosting",
        GradientBoostingClassifier::new()
            .n_estimators(100)
            .max_depth(5)
            .learning_rate(0.1),
        data
    );
    run_cv!(
        "Logistic Regression",
        LogisticRegression::new()
            .max_iter(500)
            .learning_rate(0.01),
        scaled
    );

    // Model comparison chart
    let model_names: Vec<&str> = results.iter().map(|r| r.name).collect();
    let model_accs: Vec<f64> = results.iter().map(|r| r.mean).collect();
    bar(&model_names, &model_accs)
        .title("5-Fold CV Accuracy by Model")
        .x_label("Model")
        .y_label("Mean Accuracy")
        .theme(theme.clone())
        .size(900.0, 500.0)
        .build()
        .save_svg(fig_dir.join("11_model_comparison.svg").to_str().unwrap())?;

    // ── Best model evaluation ────────────────────────────────────
    println!("\n── GRADIENT BOOSTING TEST EVALUATION ──");
    let mut gbt = GradientBoostingClassifier::new()
        .n_estimators(100)
        .max_depth(5)
        .learning_rate(0.1);
    gbt.fit(&train)?;
    let test_rows = to_row_major(&test);
    let preds = gbt.predict(&test_rows)?;
    let probas = gbt.predict_proba(&test_rows)?;

    let report = classification_report(&test.target, &preds);
    println!("{report}");

    // Confusion matrix
    let cm = confusion_matrix(&test.target, &preds);
    cm.figure()
        .save_svg(fig_dir.join("12_confusion_matrix.svg").to_str().unwrap())?;

    // Classification report bar chart
    report
        .figure()
        .save_svg(fig_dir.join("13_classification_report.svg").to_str().unwrap())?;

    // ROC curve
    let scores: Vec<f64> = probas
        .iter()
        .map(|p| p.get(1).copied().unwrap_or(0.0))
        .collect();
    let roc = roc_curve(&test.target, &scores);
    println!("ROC AUC: {:.4}", roc.auc);
    roc.roc_figure()
        .save_svg(fig_dir.join("14_roc_curve.svg").to_str().unwrap())?;

    // ── Feature importance (THE ICP) ─────────────────────────────
    println!("\n── PERMUTATION FEATURE IMPORTANCE (= ICP) ──");
    println!("  (Higher = more important for predicting engagement)\n");

    let pi = permutation_importance(
        &test.features,
        &test.target,
        &|feats: &[Vec<f64>]| {
            let n = feats[0].len();
            let m = feats.len();
            let rows: Vec<Vec<f64>> = (0..n)
                .map(|i| (0..m).map(|j| feats[j][i]).collect())
                .collect();
            gbt.predict(&rows).unwrap()
        },
        accuracy,
        5,
        42,
    );

    let mut feat_imp: Vec<(String, f64)> = data
        .feature_names
        .iter()
        .zip(pi.importances_mean.iter())
        .map(|(name, &imp)| (name.clone(), imp))
        .collect();
    feat_imp.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Print top 20
    for (i, (name, imp)) in feat_imp.iter().enumerate().take(20) {
        println!("  {:>3}. {:30} {imp:.4}", i + 1, name);
    }

    // Feature importance chart (top 20)
    let top_n = 20.min(feat_imp.len());
    let fi_names: Vec<&str> = feat_imp[..top_n].iter().map(|(n, _)| n.as_str()).collect();
    let fi_vals: Vec<f64> = feat_imp[..top_n].iter().map(|(_, v)| *v).collect();
    bar(&fi_names, &fi_vals)
        .title("Feature Importance — What Predicts Engagement (ICP)")
        .x_label("Feature")
        .y_label("Mean Accuracy Decrease")
        .theme(theme.clone())
        .size(1100.0, 600.0)
        .build()
        .save_svg(fig_dir.join("15_feature_importance_icp.svg").to_str().unwrap())?;

    // ── Also train Random Forest for comparison ──────────────────
    println!("\n── RANDOM FOREST TEST EVALUATION ──");
    let mut rf = RandomForestClassifier::new()
        .n_estimators(100)
        .max_depth(10)
        .seed(42);
    rf.fit(&train)?;
    let preds_rf = rf.predict(&test_rows)?;
    let report_rf = classification_report(&test.target, &preds_rf);
    println!("{report_rf}");

    // ── Logistic Regression (for coefficient interpretation) ─────
    println!("\n── LOGISTIC REGRESSION TEST EVALUATION ──");
    let mut lr = LogisticRegression::new().max_iter(500).learning_rate(0.01);
    lr.fit(&train_s)?;
    let test_s_rows = to_row_major(&test_s);
    let preds_lr = lr.predict(&test_s_rows)?;
    let report_lr = classification_report(&test_s.target, &preds_lr);
    println!("{report_lr}");

    // ═══════════════════════════════════════════════════════════════
    //  Generate HTML Dashboard
    // ═══════════════════════════════════════════════════════════════
    let svg_files = [
        ("Pipeline Status Distribution", "01_status_funnel.svg"),
        ("City Tier Distribution", "02_city_tier_pie.svg"),
        ("Lead Volume by Source", "03_source_volume.svg"),
        ("Reply Rate by Source", "04_source_reply_rate.svg"),
        ("Researcher Performance", "05_researcher_performance.svg"),
        ("Month Seasonality", "06_month_seasonality.svg"),
        (
            "Outreach Diminishing Returns",
            "07_outreach_diminishing_returns.svg",
        ),
        ("Engagement: Sent vs Received", "08_engagement_scatter.svg"),
        ("Bid Outcomes", "09_bid_outcomes.svg"),
        ("Reply Rate by City Tier", "10_tier_reply_rate.svg"),
        ("Model Comparison (CV)", "11_model_comparison.svg"),
        ("Confusion Matrix", "12_confusion_matrix.svg"),
        ("Classification Report", "13_classification_report.svg"),
        ("ROC Curve", "14_roc_curve.svg"),
        ("Feature Importance (ICP)", "15_feature_importance_icp.svg"),
        ("Leads Found per Year", "16_yoy_leads_volume.svg"),
        ("Reply Rate by Year", "17_yoy_reply_rate.svg"),
        ("Bids Placed by Year", "18_yoy_bids.svg"),
        ("Lead Source Mix by Year", "19_yoy_source_mix.svg"),
        ("Researcher Activity by Year", "20_yoy_researcher_activity.svg"),
        ("City Tier Composition by Year", "21_yoy_city_tier.svg"),
        (
            "Reply Rate: Major vs International Trend",
            "22_yoy_tier_reply_trend.svg",
        ),
        (
            "Conferences by Target Year (Engagement)",
            "23_conf_year_engagement.svg",
        ),
        ("Cumulative Pipeline Growth", "24_cumulative_growth.svg"),
    ];

    let mut html = String::from(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sales Pipeline Analysis &amp; ICP Report</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 2rem; background: #f8f9fa; }
  h1 { color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 0.5rem; }
  h2 { color: #16213e; margin-top: 2rem; }
  .chart { background: white; border-radius: 8px; padding: 1rem; margin: 1.5rem 0;
           box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center; }
  .chart img { max-width: 100%; height: auto; }
  .section { margin: 3rem 0; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
  @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1>Sales Pipeline Analysis &amp; ICP Report</h1>
<p>Generated from 8,539 conference leads (2015–2026) | Target: reply engagement prediction</p>
"#,
    );

    let sections = [
        (
            "Pipeline Overview",
            vec![
                "01_status_funnel.svg",
                "02_city_tier_pie.svg",
                "09_bid_outcomes.svg",
                "10_tier_reply_rate.svg",
            ],
        ),
        (
            "Lead Sources &amp; Outreach",
            vec![
                "03_source_volume.svg",
                "04_source_reply_rate.svg",
                "07_outreach_diminishing_returns.svg",
                "08_engagement_scatter.svg",
            ],
        ),
        (
            "Team Performance",
            vec![
                "05_researcher_performance.svg",
                "06_month_seasonality.svg",
            ],
        ),
        (
            "Year-over-Year Historical Trends",
            vec![
                "16_yoy_leads_volume.svg",
                "17_yoy_reply_rate.svg",
                "18_yoy_bids.svg",
                "19_yoy_source_mix.svg",
                "20_yoy_researcher_activity.svg",
                "21_yoy_city_tier.svg",
                "22_yoy_tier_reply_trend.svg",
                "23_conf_year_engagement.svg",
                "24_cumulative_growth.svg",
            ],
        ),
        (
            "Machine Learning — ICP Model",
            vec![
                "11_model_comparison.svg",
                "15_feature_importance_icp.svg",
                "12_confusion_matrix.svg",
                "13_classification_report.svg",
                "14_roc_curve.svg",
            ],
        ),
    ];

    for (section_title, filenames) in &sections {
        html.push_str(&format!(
            r#"<div class="section">
<h2>{section_title}</h2>
<div class="grid">
"#
        ));
        for filename in filenames {
            // Find title from svg_files
            let title = svg_files
                .iter()
                .find(|(_, f)| f == filename)
                .map(|(t, _)| *t)
                .unwrap_or(filename);
            html.push_str(&format!(
                r#"<div class="chart">
<h3>{title}</h3>
<img src="figures/{filename}" alt="{title}">
</div>
"#
            ));
        }
        html.push_str("</div>\n</div>\n");
    }

    html.push_str("</body>\n</html>");

    let html_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("report")
        .join("dashboard.html");
    std::fs::write(&html_path, &html)?;

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  ALL DONE");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Charts saved to: {}", fig_dir.display());
    println!("Dashboard: {}", html_path.display());

    Ok(())
}
