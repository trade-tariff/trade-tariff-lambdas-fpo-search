let
  pkgs = import <nixpkgs> {};
  stdenv = pkgs.stdenv;
in pkgs.mkShell {
  LD_LIBRARY_PATH="${stdenv.cc.cc.lib}/lib/";
  buildInputs = [
    (pkgs.python311.withPackages (ps: [
      ps.torch
      ps.sentence-transformers
      ps.sentry-sdk
      ps.toml
      ps.typing-extensions
    ]))
    pkgs.nodejs_21
  ];
  shellHook = ''
    source venv/bin/activate
  '';
}
