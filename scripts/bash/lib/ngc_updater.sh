#!/bin/bash
# ===========================================================================
#  ngc_updater.sh — Fetch and update NGC compatibility matrix
#
#  Scrapes the official NVIDIA Triton compatibility page and merges new
#  entries into the local ngc_matrix.conf.
#
#  Functions: update_ngc_matrix
#  Depends:   lib/logging.sh, curl, python3 (for HTML parsing)
#
#  The scraper is best-effort: if the page structure changes or the
#  network is unreachable, the existing conf file is left untouched.
# ===========================================================================

[[ -n "${_LIB_NGC_UPDATER_LOADED:-}" ]] && return 0
_LIB_NGC_UPDATER_LOADED=1

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_LIB_DIR}/logging.sh"

_COMPAT_URL="https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/introduction/compatibility.html"

# ---------------------------------------------------------------------------
#  _scrape_compat_matrix
#  Downloads the compatibility page and extracts trtllm-python-py3 rows
#  into the conf format.  Outputs lines to stdout.
#  Returns 1 on any failure (network, parse, empty result).
# ---------------------------------------------------------------------------
_scrape_compat_matrix() {
    local html
    html=$(curl -fsSL --connect-timeout 15 --max-time 30 "$_COMPAT_URL" 2>/dev/null) \
        || { log_warn "Failed to fetch compatibility page"; return 1; }

    if [ -z "$html" ]; then
        log_warn "Empty response from compatibility page"
        return 1
    fi

    python3 - "$html" <<'PYEOF' || return 1
import sys, re
from html.parser import HTMLParser

html = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()

class TritonTableParser(HTMLParser):
    """Extract rows from the trtllm-python-py3 table."""

    def __init__(self):
        super().__init__()
        self.in_trtllm_section = False
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.current_row = []
        self.cell_text = ""
        self.rows = []
        self.header_seen = False
        self.tag_stack = []

    def handle_starttag(self, tag, attrs):
        self.tag_stack.append(tag)
        if tag == "h2":
            self.in_trtllm_section = False
        if tag == "table" and self.in_trtllm_section:
            self.in_table = True
        if tag == "tr" and self.in_table:
            self.in_row = True
            self.current_row = []
        if tag in ("td", "th") and self.in_row:
            self.in_cell = True
            self.cell_text = ""

    def handle_endtag(self, tag):
        if self.tag_stack:
            self.tag_stack.pop()
        if tag in ("td", "th") and self.in_cell:
            self.in_cell = False
            self.current_row.append(self.cell_text.strip())
        if tag == "tr" and self.in_row:
            self.in_row = False
            if self.current_row:
                if not self.header_seen:
                    self.header_seen = True
                else:
                    self.rows.append(self.current_row)
        if tag == "table" and self.in_table:
            self.in_table = False

    def handle_data(self, data):
        if self.in_cell:
            self.cell_text += data
        stripped = data.strip()
        if "trtllm-python-py3" in stripped.lower():
            if "h2" in self.tag_stack or "h3" in self.tag_stack:
                self.in_trtllm_section = True
                self.header_seen = False

parser = TritonTableParser()
parser.feed(html)

if not parser.rows:
    sys.exit(1)

for row in parser.rows:
    if len(row) < 9:
        continue

    # Columns: Triton release | NGC Tag | Python | Torch | TensorRT |
    #          TensorRT-LLM | CUDA | CUDA Driver | Size
    ngc_tag_full = row[1].strip()
    python_ver = row[2].strip()
    trtllm_ver = row[5].strip()
    cuda_ver = row[6].strip()
    driver_ver = row[7].strip()
    size_raw = row[8].strip()

    # Extract NGC tag: "nvcr.io/nvidia/tritonserver:25.08-trtllm-python-py3" -> "25.08"
    m = re.search(r':(\d+\.\d+)-trtllm', ngc_tag_full)
    if not m:
        continue
    ngc_tag = m.group(1)

    # Normalize python version: "Python 3.12.3" -> "3.12"
    pm = re.search(r'(\d+\.\d+)', python_ver)
    py_ver = pm.group(1) if pm else "-"

    # Normalize driver: take major.minor (e.g. "575.51.03" -> "575.51")
    dm = re.match(r'(\d+\.\d+)', driver_ver)
    min_driver = dm.group(1) if dm else driver_ver

    # Normalize CUDA: take major.minor (e.g. "12.9.0.043" -> "12.9")
    cm = re.match(r'(\d+\.\d+)', cuda_ver)
    cuda = cm.group(1) if cm else cuda_ver

    # Normalize TRT-LLM version: strip suffixes like ".post1"
    trtllm = re.sub(r'\.post\d+$', '', trtllm_ver)

    # Normalize size: "20.49 GB" -> "20.49", "18.3G" -> "18.3"
    sm = re.search(r'([\d.]+)\s*G', size_raw)
    size = sm.group(1) if sm else "-"

    print(f"{ngc_tag}  {min_driver}  {trtllm}  {cuda}  {py_ver}  {size}")
PYEOF
}

# ---------------------------------------------------------------------------
#  _merge_matrix <scraped_file> <existing_conf>
#  Merges scraped entries into the existing conf file.
#  - New tags are added
#  - Existing tags are updated if scraped data has more info (e.g. size)
#  - Tags only in conf (not on website) are preserved
#  - Result is sorted newest-first
#  Outputs merged content to stdout.
# ---------------------------------------------------------------------------
_merge_matrix() {
    local scraped_file="$1"
    local existing_conf="$2"

    python3 - "$scraped_file" "$existing_conf" <<'PYEOF'
import sys, re
from collections import OrderedDict

def parse_conf(path):
    """Parse conf file into {tag: (line, fields)} preserving comments."""
    entries = OrderedDict()
    comments = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                comments.append(line.rstrip())
                continue
            parts = stripped.split()
            if len(parts) >= 4:
                tag = parts[0]
                entries[tag] = parts
    return comments, entries

def tag_sort_key(tag):
    """Sort key for NGC tags: 26.02 > 25.11 > 25.08 etc."""
    parts = tag.split(".")
    return (int(parts[0]), int(parts[1]))

scraped_file, existing_conf = sys.argv[1], sys.argv[2]

# Parse scraped data
scraped = OrderedDict()
with open(scraped_file) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 4:
            scraped[parts[0]] = parts

# Parse existing conf
comments, existing = parse_conf(existing_conf)

# Merge: scraped wins for fields that are "-" in existing
merged = OrderedDict()
all_tags = set(list(scraped.keys()) + list(existing.keys()))

for tag in all_tags:
    s = scraped.get(tag)
    e = existing.get(tag)

    if s and e:
        # Merge: prefer non-"-" values
        result = list(e)
        # Pad both to 6 fields
        while len(result) < 6:
            result.append("-")
        s_padded = list(s)
        while len(s_padded) < 6:
            s_padded.append("-")

        for i in range(len(result)):
            if result[i] == "-" and i < len(s_padded) and s_padded[i] != "-":
                result[i] = s_padded[i]
            # Also update size if scraped has a real value
            if i == 5 and s_padded[i] != "-":
                result[i] = s_padded[i]
        merged[tag] = result
    elif s:
        while len(s) < 6:
            s.append("-")
        merged[tag] = s
    else:
        while len(e) < 6:
            e.append("-")
        merged[tag] = e

# Sort newest-first
sorted_tags = sorted(merged.keys(), key=tag_sort_key, reverse=True)

# Output: comments then sorted entries
for c in comments:
    # Update the "Last updated" line
    if "Last updated" in c:
        from datetime import date
        c = f"#  Last updated: {date.today().isoformat()} (update-matrix)"
    print(c)

for tag in sorted_tags:
    fields = merged[tag]
    # Format with consistent spacing
    print(f"{fields[0]:6s} {fields[1]:8s} {fields[2]:8s} {fields[3]:5s} {fields[4]:5s} {fields[5]}")
PYEOF
}

# ---------------------------------------------------------------------------
#  update_ngc_matrix [conf_path]
#  Main entry point.  Fetches the latest data from the NVIDIA compatibility
#  page and merges it into the local matrix conf file.
#  The existing file is never overwritten on failure.
# ---------------------------------------------------------------------------
update_ngc_matrix() {
    local conf_path="${1:-}"

    if [ -z "$conf_path" ]; then
        local script_dir
        script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
        conf_path="${script_dir}/ngc_matrix.conf"
    fi

    log_step "Updating NGC compatibility matrix"
    log_info "Source: $_COMPAT_URL"
    log_info "Target: $conf_path"

    if [ ! -f "$conf_path" ]; then
        log_error "Matrix conf not found: $conf_path"
        log_error "Expected at: scripts/bash/ngc_matrix.conf"
        return 1
    fi

    local tmp_scraped
    tmp_scraped=$(mktemp /tmp/ngc_scraped.XXXXXX) || return 1

    if _scrape_compat_matrix > "$tmp_scraped" 2>/dev/null; then
        local count
        count=$(wc -l < "$tmp_scraped" | tr -d ' ')
        if [ "$count" -eq 0 ]; then
            log_warn "Scraper returned no entries — page structure may have changed"
            log_info "Existing matrix preserved unchanged"
            rm -f "$tmp_scraped"
            return 1
        fi
        log_info "Scraped $count entries from official page"
    else
        log_warn "Failed to scrape compatibility page (network or parse error)"
        log_info "Existing matrix preserved unchanged"
        rm -f "$tmp_scraped"
        return 1
    fi

    local tmp_merged
    tmp_merged=$(mktemp /tmp/ngc_merged.XXXXXX) || { rm -f "$tmp_scraped"; return 1; }

    if _merge_matrix "$tmp_scraped" "$conf_path" > "$tmp_merged" 2>/dev/null; then
        local new_count old_count
        new_count=$(grep -v '^\s*#' "$tmp_merged" | grep -v '^\s*$' | wc -l | tr -d ' ')
        old_count=$(grep -v '^\s*#' "$conf_path" | grep -v '^\s*$' | wc -l | tr -d ' ')

        cp "$conf_path" "${conf_path}.bak"
        mv "$tmp_merged" "$conf_path"

        if [ "$new_count" -gt "$old_count" ]; then
            local added=$((new_count - old_count))
            log_info "Matrix updated: $old_count → $new_count entries (+$added new)"
        elif [ "$new_count" -eq "$old_count" ]; then
            log_info "Matrix up to date ($new_count entries, sizes/versions may have been refreshed)"
        else
            log_warn "Matrix shrank: $old_count → $new_count (check ${conf_path}.bak)"
        fi

        log_info "Backup saved: ${conf_path}.bak"
    else
        log_warn "Merge failed — existing matrix preserved"
        rm -f "$tmp_merged"
    fi

    rm -f "$tmp_scraped"
}
