## Run all protocols

Run on one terminal for Player0
```sh
bazel run -c opt examples/cpp/cli:main -- --rank 0 --prot All
```

Run on other terminal for Player1
```sh
bazel run -c opt examples/cpp/cli:main -- --rank 1 --prot All
```
## Run a specific protocol

```sh
USAGE: main [options]

OPTIONS:

Color Options:

  --color            - Use colors in output (default=autodetect)

General options:

  --b0=<int>         - Bitwidth for P0 (1 <= b0 <= 128)
  --b1=<int>         - Bitwidth for P1 (1 <= b1 <= 128)
  --parties=<string> - server list, format: host1:port1[,host2:port2, ...]
  --prot=<string>    - Protocol:
                       All (run all protocols)
                       OLE2k (oblivious linear evaluation over 2^k)
                       OLEp (oblivious linear evaluation over p)
                       NExp (negative exponent)
                       CMP (comparison)
                       R2R (ring-to-ring)
                       R2F (ring-to-field)
                       F2R (field-to-ring)
                       TRC2k (truncate over Z2k)
  --rank=<int>       - Rank. 0 for P0, and 1 for P1

Generic Options:

  --help             - Display available options (--help-hidden for more)
  --help-list        - Display list of available options (--help-list-hidden for more)
  --version          - Display the version of this program
```
