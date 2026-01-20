// runner.cpp
// Build: g++ -O2 -std=c++17 runner.cpp -o runner
// Edit the lists in main(): binaries, files, configs, output_csv

#include <bits/stdc++.h>

#if defined(_WIN32)
  #include <windows.h>
  #define popen _popen
  #define pclose _pclose
#endif


static std::string csv_escape(const std::string& s) {
    bool need_quotes = false;
    for (char c : s) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') { need_quotes = true; break; }
    }
    if (!need_quotes) return s;

    std::string out = "\"";
    for (char c : s) out += (c == '"') ? "\"\"" : std::string(1, c);
    out += "\"";
    return out;
}

// Extract first numeric value from stdout (CF). If none, parse fails.
static std::pair<bool, double> parse_first_number(const std::string& s) {
    static const std::regex re(R"(([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?))");
    std::smatch m;
    if (std::regex_search(s, m, re)) {
        try { return {true, std::stod(m.str(1))}; }
        catch (...) { return {false, 0.0}; }
    }
    return {false, 0.0};
}

// Run command, capture stdout, measure wall time ms.
static std::string run_and_capture_stdout(const std::string& cmd, int& exit_code, double& ms) {
    using clock = std::chrono::steady_clock;

    auto t0 = clock::now();
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        exit_code = -1;
        ms = 0.0;
        return "";
    }

    std::string output;
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) output += buffer;

    int rc = pclose(pipe);
    #if defined(_WIN32)
    exit_code = rc;
    #else
    if (WIFEXITED(rc)) exit_code = WEXITSTATUS(rc);
    else exit_code = rc;
    #endif
    
    auto t1 = clock::now();
    assert(rc == 0);
    ms = std::chrono::duration<double>(t1 - t0).count() * 1000;
    return output;
}

int main() {
    // =========================
    // FILL THESE LISTS MANUALLY
    // =========================

    // One or more binaries to benchmark (paths to executables).
    // If you only have one binary, just keep a single entry.
    const std::vector<std::string> configs = {
        "fsst",
        "btrfsst"
        // "/path/to/another_binary"
    };

    // Input files (one file per run).
    std::vector<std::string> files;
    namespace fs = std::filesystem;
    std::string path = "../../data/refined";
    
    for (const auto& entry : fs::recursive_directory_iterator(path)) {
        if(entry.is_regular_file())
            files.push_back(entry.path().string());
    }
 

    // Output CSV path
    const std::string output_csv = "../csv/results_py.csv";


    std::ofstream out(output_csv);
    if (!out) {
        std::cerr << "Failed to open output CSV: " << output_csv << "\n";
        return 2;
    }

    // CSV header: includes binary too (helpful if you run multiple binaries)
    out << "configuration,Time,CF\n";

    for (const auto& cfg : configs) {
        for (const auto& file : files) {
            std::ostringstream cmd;
            cmd << "python3";
            cmd << " ./" << cfg << ".py";
            cmd << " " << file;
            int exit_code = 0;
            double ms = 0.0;
            std::string stdout_text = run_and_capture_stdout(cmd.str(), exit_code, ms);
            if(ms >= 1000) {
                std::cout << file << std::endl;
            }

            auto [ok, cf_value] = parse_first_number(stdout_text);

            // If CF isn't numeric, store raw stdout (escaped) in CF column.
            std::string cf_field = ok ? std::to_string(cf_value)
                                        : csv_escape(stdout_text);

            out << cfg << ","
                << ms << ","
                << cf_field << "\n";
        }
    }

    std::cerr << "Wrote: " << output_csv << "\n";
    return 0;
}
