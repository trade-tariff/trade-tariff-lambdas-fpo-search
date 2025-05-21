{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        stdenv = pkgs.stdenv;
        pkgs = import nixpkgs {
          system = system;
          config.allowUnfree = true;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          LD_LIBRARY_PATH = "${stdenv.cc.cc.lib}/lib/";
          shellHook = ''
            if [[ ! -f venv ]]
            then
              python -m venv venv
            fi
          '';

          buildInputs = [
            pkgs.python311
            pkgs.python311Packages.pip
            pkgs.nodejs_latest
            pkgs.yarn
            pkgs.rufo
            pkgs.ruff
          ];
        };
      });
}
