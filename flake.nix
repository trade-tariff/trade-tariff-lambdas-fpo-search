{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    pre-commit-hooks.url = "github:cachix/git-hooks.nix";
  };
  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      pre-commit-hooks,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          system = system;
          config.allowUnfree = true;
        };
        # Keep native hooks on the Ruff release declared in .pre-commit-config.yaml.
        ruff =
          (import (builtins.fetchTree {
            type = "github";
            owner = "NixOS";
            repo = "nixpkgs";
            rev = "2ac2235f3fdb7cff497dbe490c4cb8497f373e29";
            narHash = "sha256-mdwWjRoMVa0g4cw7kqFBO3MI17Cg0KtlYPTAV4NQQ4Y=";
          }) { inherit system; }).ruff;

        reviewEnvironment = ''
          if [ "''${REVIEW_SANDBOX:-}" != 1 ] || [ "$PWD" != /workspace ]; then exit 64; fi
          export TMPDIR=/workspace/.review/tmp VIRTUAL_ENV=/workspace/.review/venv PRE_COMMIT_HOME=/workspace/.review/pre-commit
          export PATH="$VIRTUAL_ENV/bin:$PATH" PIP_CONFIG_FILE=/dev/null PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1
          unset PYTHONPATH PYTHONHOME
          export AWS_EC2_METADATA_DISABLED=true AWS_CONFIG_FILE=/dev/null AWS_SHARED_CREDENTIALS_FILE=/dev/null
          export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
          export LD_LIBRARY_PATH="${
            pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
            ]
          }"
          mkdir -p "$TMPDIR"
        '';
        reviewTools = import (builtins.fetchTree {
          type = "github";
          owner = "NixOS";
          repo = "nixpkgs";
          rev = "4a29d733e8a7d5b824c3d8c958a946a9867b3eb2";
          narHash = "sha256-f+JACbTqzZ+G92DSnXOUGRhGANb8Blh7CoeYOeBF8/U=";
        }) { inherit system; };
        reviewHook =
          id: entry: attrs:
          {
            inherit id entry;
            name = id;
            language = "system";
          }
          // attrs;
        reviewPreCommitConfig = pkgs.writeText "review-pre-commit.json" (
          builtins.toJSON {
            repos = [
              {
                repo = "local";
                hooks = [
                  (reviewHook "ruff-check" "${ruff}/bin/ruff check --force-exclude" {
                    types_or = [
                      "python"
                      "pyi"
                      "jupyter"
                    ];
                    require_serial = true;
                  })
                  (reviewHook "ruff-format" "${ruff}/bin/ruff format --force-exclude" {
                    types_or = [
                      "python"
                      "pyi"
                      "jupyter"
                    ];
                    require_serial = true;
                  })
                  (reviewHook "trailing-whitespace"
                    "${reviewTools.python3Packages.pre-commit-hooks}/bin/trailing-whitespace-fixer"
                    {
                      types = [ "text" ];
                      exclude = "reference_data\\/.*";
                      stages = [
                        "pre-commit"
                        "pre-push"
                        "manual"
                      ];
                    }
                  )
                  (reviewHook "end-of-file-fixer"
                    "${reviewTools.python3Packages.pre-commit-hooks}/bin/end-of-file-fixer"
                    {
                      types = [ "text" ];
                      exclude = "reference_data\\/.*";
                      stages = [
                        "pre-commit"
                        "pre-push"
                        "manual"
                      ];
                    }
                  )
                  (reviewHook "check-yaml" "${reviewTools.python3Packages.pre-commit-hooks}/bin/check-yaml" {
                    types = [ "yaml" ];
                    exclude = "reference_data\\/.*";
                  })
                  (reviewHook "check-merge-conflict"
                    "${reviewTools.python3Packages.pre-commit-hooks}/bin/check-merge-conflict"
                    {
                      types = [ "text" ];
                      exclude = "reference_data\\/.*";
                    }
                  )
                  (reviewHook "shellcheck" "${reviewTools.shellcheck}/bin/shellcheck" {
                    types = [ "shell" ];
                    args = [ "--severity=warning" ];
                  })
                  (reviewHook "markdownlint-fix" "${reviewTools.markdownlint-cli}/bin/markdownlint --fix" {
                    types = [ "markdown" ];
                    args = [
                      "--ignore"
                      "terraform"
                    ];
                  })
                  (reviewHook "trufflehog"
                    "${reviewTools.trufflehog}/bin/trufflehog git file://. --since-commit HEAD --results=verified --fail --trust-local-git-config"
                    { pass_filenames = false; }
                  )
                  (reviewHook "actionlint-docker" "${reviewTools.actionlint}/bin/actionlint" {
                    types = [ "yaml" ];
                    files = "^\\.github/workflows/";
                  })
                ];
              }
            ];
          }
        );

        pre-commit-check = pre-commit-hooks.lib.${system}.run {
          src = ./.;
          configPath = ".pre-commit-config-nix.yaml";
          hooks = {
            ruff-check = {
              enable = true;
              package = ruff;
              entry = "${ruff}/bin/ruff check --force-exclude";
              types_or = [
                "python"
                "pyi"
                "jupyter"
              ];
              require_serial = true;
            };
            ruff-format = {
              enable = true;
              package = ruff;
              entry = "${ruff}/bin/ruff format --force-exclude";
              types = [ "file" ];
              types_or = [
                "python"
                "pyi"
                "jupyter"
              ];
              require_serial = true;
            };
            trim-trailing-whitespace = {
              enable = true;
              excludes = [ "reference_data/.*" ];
            };
            end-of-file-fixer = {
              enable = true;
              excludes = [ "reference_data/.*" ];
            };
            check-yaml = {
              enable = true;
              excludes = [ "reference_data/.*" ];
            };
            check-merge-conflicts = {
              enable = true;
              excludes = [ "reference_data/.*" ];
            };
            shellcheck = {
              enable = true;
              package = pkgs.shellcheck;
              entry = "${pkgs.shellcheck}/bin/shellcheck --severity=warning";
            };
            markdownlint = {
              enable = true;
              package = pkgs.markdownlint-cli;
              entry = "${pkgs.markdownlint-cli}/bin/markdownlint --fix";
              excludes = [ "^terraform/" ];
            };
            trufflehog = {
              enable = true;
              package = pkgs.trufflehog;
              entry = "${pkgs.trufflehog}/bin/trufflehog git file://. --since-commit HEAD --fail";
            };
            actionlint = {
              enable = true;
              package = pkgs.actionlint;
              entry = "${pkgs.actionlint}/bin/actionlint";
            };
          };
        };

        baseInputs = [
          pkgs.python313
          pkgs.python313Packages.pandas
          pkgs.python313Packages.pip
          pkgs.nodejs_latest
          pkgs.yarn
          ruff
          pkgs.zlib
        ];

        baseShellHook = ''
          ${pre-commit-check.shellHook}
          if [ ! -d "venv" ]; then
            ${pkgs.python313}/bin/python -m venv venv
            source venv/bin/activate
            pip install --upgrade pip
            pip install -r requirements.txt  # CPU-only install here
          fi
          source venv/bin/activate
        '';
      in
      {
        devShells = {
          review = pkgs.mkShell {
            shellHook = "";
            PRE_COMMIT_NO_CONCURRENCY = "1";
            GOMAXPROCS = "2";
            packages = [
              pkgs.gitMinimal
              (pkgs.python313.withPackages (ps: [
                ps.pip
                ps.virtualenv
              ]))
              pkgs.pre-commit
            ];
            nativeBuildInputs = [
              (pkgs.writeShellApplication {
                name = "review-bootstrap";
                text = reviewEnvironment + ''
                  if [ ! -s /deps/python-inputs/locked-requirements.txt ]; then echo 'Blocked: verified CPU wheels for the declared requirements are unavailable.' >&2; exit 78; fi
                  (cd /deps/python-inputs; sha256sum -c checksums)
                  ${
                    pkgs.python313.withPackages (ps: [
                      ps.pip
                      ps.virtualenv
                    ])
                  }/bin/python -m virtualenv --no-download "$VIRTUAL_ENV"
                  "$VIRTUAL_ENV/bin/python" -m pip install --no-index --find-links=/deps/python-inputs/wheels --require-hashes -r /deps/python-inputs/locked-requirements.txt
                  "$VIRTUAL_ENV/bin/python" -m pip check
                '';
              })
              (pkgs.writeShellApplication {
                name = "review-check";
                text = reviewEnvironment + ''
                  status=0
                  pre-commit run --all-files --config ${reviewPreCommitConfig} || status=$?
                  python - <<'PY' || status=$?
                  import json
                  import sys
                  import unittest
                  suite = unittest.defaultTestLoader.discover('.')
                  result = unittest.TextTestRunner(verbosity=2, buffer=True).run(suite)
                  print('FULL_SUITE_SUMMARY=' + json.dumps({'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped)}))
                  sys.exit(not result.wasSuccessful() or result.testsRun == 0 or bool(result.skipped))
                  PY
                  exit "$status"
                '';
              })
            ];
            TZDIR = "${pkgs.tzdata.outPath}/share/zoneinfo";
            SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
          };
          default = pkgs.mkShell {
            inherit baseShellHook;
            buildInputs = baseInputs;
          };
          cuda = pkgs.mkShell {
            CUDA_HOME = "${pkgs.cudaPackages.cudatoolkit}";
            PATH = "${pkgs.cudaPackages.cudatoolkit}/bin:$PATH";
            shellHook = ''
              export LD_LIBRARY_PATH="${
                pkgs.lib.makeLibraryPath [
                  pkgs.stdenv.cc.cc
                  pkgs.cudaPackages.cudatoolkit
                  pkgs.cudaPackages.cudnn
                  pkgs.cudaPackages.libcublas
                  pkgs.cudaPackages.libcufft
                  pkgs.cudaPackages.libnvjitlink
                ]
              }:/run/opengl-driver/lib:$LD_LIBRARY_PATH"
              ${pre-commit-check.shellHook}
              if [ ! -d "venv" ]; then
                ${pkgs.python313}/bin/python -m venv venv
                source venv/bin/activate
                pip install --upgrade pip
                pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
              fi
              source venv/bin/activate
            '';

            buildInputs = baseInputs ++ [
              pkgs.cudaPackages.cudatoolkit
              pkgs.cudaPackages.cudnn
              pkgs.cudaPackages.libcublas
              pkgs.cudaPackages.libcufft
              pkgs.cudaPackages.libnvjitlink
            ];
          };
        };
      }
    );
}
