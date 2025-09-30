{ pkgs, lib, config, ... }:

{
  # Python via Nix (no pip build pain)
  languages.python.enable = true;
  languages.python.package = pkgs.python311;
  languages.python.venv.enable = false;

  packages = [
    (pkgs.python311.withPackages (ps: with ps; [
      requests
      pandas
      numpy
      tqdm
      openpyxl   # needed for .xlsx
      xlrd       # needed for legacy .xls
      python-dotenv
      matplotlib
    ]))
    pkgs.fish
    pkgs.git
    pkgs.curl
    pkgs.jq
  ];

  dotenv.enable = true;
  dotenv.filename = ".env";

  scripts.run-swaps.exec = ''
    python helius_swaps_min.py \
      --addresses-csv addresses.csv \
      --output-csv tdccp_swaps.csv --strict
  '';

  scripts.check-keys.exec = ''
    test -n "$HELIUS_API_KEY" && echo "HELIUS_API_KEY present" || { echo "HELIUS_API_KEY missing"; exit 1; }
  '';

  enterShell = ''
    echo "Python: $(python --version)"
    python - <<'PY'
try:
  import pandas, openpyxl, xlrd
  print("pandas:", pandas.__version__)
  print("openpyxl:", openpyxl.__version__)
  print("xlrd:", xlrd.__version__)
except Exception as e:
  print("Problem:", e)
PY
  '';
}

