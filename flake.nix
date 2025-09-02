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
        pre-commit-check = pre-commit-hooks.lib.${system}.run {
          src = ./.;
          configPath = ".pre-commit-config-nix.yaml";
          hooks = {
            ruff-check = {
              enable = true;
              entry = "${pkgs.ruff}/bin/ruff check --force-exclude";
              types_or = [
                "python"
                "pyi"
                "jupyter"
              ];
              require_serial = true;
            };

            ruff-format = {
              enable = true;
              entry = "${pkgs.ruff}/bin/ruff format --force-exclude";
              types_or = [
                "python"
                "pyi"
                "jupyter"
              ];
              require_serial = true;
            };

            trim-trailing-whitespace.enable = true;
            end-of-file-fixer.enable = true;
            check-yaml.enable = true;
            shellcheck = {
              enable = true;
              package = pkgs.shellcheck;
              entry = "${pkgs.shellcheck}/bin/shellcheck --severity=warning";
            };
            markdownlint = {
              enable = true;
              package = pkgs.markdownlint-cli;
              entry = "${pkgs.markdownlint-cli}/bin/markdownlint --fix";
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
      in
      {
        devShells = {
          default = pkgs.mkShell {
            CUDA_HOME = "${pkgs.cudaPackages_12_8.cudatoolkit}";
            PATH = "${pkgs.cudaPackages_12_8.cudatoolkit}/bin:$PATH";

            shellHook = ''
              ${pre-commit-check.shellHook}

              export LD_LIBRARY_PATH="${
                pkgs.lib.makeLibraryPath [
                  pkgs.stdenv.cc.cc
                  pkgs.cudaPackages_12_8.cudatoolkit
                  pkgs.cudaPackages_12_8.cudnn
                  pkgs.cudaPackages_12_8.libcublas
                  pkgs.cudaPackages_12_8.libcufft
                  pkgs.cudaPackages_12_8.libnvjitlink
                ]
              }:/run/opengl-driver/lib:$LD_LIBRARY_PATH";

              if [ ! -d "venv" ]; then
                ${pkgs.python313}/bin/python -m venv venv
                source venv/bin/activate
                pip install --upgrade pip
                pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
              fi
              source .venv/bin/activate
            '';

            buildInputs = [
              pkgs.python313
              pkgs.python313Packages.pandas
              pkgs.python313Packages.pip
              pkgs.nodejs_latest
              pkgs.yarn
              pkgs.ruff
              pkgs.zlib
              pkgs.cudaPackages_12_8.cudatoolkit
              pkgs.cudaPackages_12_8.cudnn
              pkgs.cudaPackages_12_8.libcublas
              pkgs.cudaPackages_12_8.libcufft
              pkgs.cudaPackages_12_8.libnvjitlink
            ];
          };
        };
      }
    );
}
