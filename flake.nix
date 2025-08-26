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
        stdenv = pkgs.stdenv;
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
              entry = "${pkgs.markdownlint-cli}/bin/mdl --fix";
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
            inherit (pre-commit-check) shellHook;

            LD_LIBRARY_PATH = "${stdenv.cc.cc.lib}/lib/";

            buildInputs = [
              pkgs.python311
              pkgs.python311Packages.pip
              pkgs.nodejs_latest
              pkgs.yarn
              pkgs.ruff
            ];
          };
        };
      }
    );
}
