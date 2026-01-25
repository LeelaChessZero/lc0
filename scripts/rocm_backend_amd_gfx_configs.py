#!/usr/bin/env python3
"""AMD GPU Architecture Configurations - Tuned configurations only"""

CONFIGS = {
    'gfx1151': {
        'name': 'Radeon 8060S',
        'family': 'RDNA3',
        'flash_attention': {
            'nthreads_d32': 256,
            'occupancy_d32': 2,
            'nbatch_fa_d32': 64,
            'nbatch_k2_d32': 24,  # Optimized tile size for balance
            'nbatch_v2_d32': 24,  # Optimized tile size for balance
            'nbatch_combine_d32': 32,
            'nstages_d32': 2,
        }
    },
}

def get_flags(arch):
    """Get flash attention compile flags."""
    if arch not in CONFIGS:
        return None

    config = CONFIGS[arch]
    fa = config['flash_attention']
    flags = [
        '-DUSE_FLASH_ATTENTION=1',
        f'-D{config["family"]}',
        f'-DFATTN_NTHREADS_D32={fa["nthreads_d32"]}',
        f'-DFATTN_OCCUPANCY_D32={fa["occupancy_d32"]}',
        f'-DFATTN_NBATCH_FA_D32={fa["nbatch_fa_d32"]}',
        f'-DFATTN_NBATCH_K2_D32={fa["nbatch_k2_d32"]}',
        f'-DFATTN_NBATCH_V2_D32={fa["nbatch_v2_d32"]}',
        f'-DFATTN_NBATCH_COMBINE_D32={fa["nbatch_combine_d32"]}',
        f'-DFATTN_NSTAGES_D32={fa["nstages_d32"]}',
    ]
    return ' '.join(flags)

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: rocm_backend_amd_gfx_configs.py <command> [arch]")
        print("Commands: list, flags <arch>, name <arch>")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == 'list':
        print("Tuned AMD GPU architectures:")
        for arch, config in CONFIGS.items():
            print(f"  {arch:10} - {config['name']}")

    elif cmd == 'flags' and len(sys.argv) >= 3:
        arch = sys.argv[2]
        flags = get_flags(arch)
        if flags:
            print(flags)
        else:
            sys.exit(1)

    elif cmd == 'name' and len(sys.argv) >= 3:
        arch = sys.argv[2]
        if arch in CONFIGS:
            print(CONFIGS[arch]['name'])
        else:
            sys.exit(1)

    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)
