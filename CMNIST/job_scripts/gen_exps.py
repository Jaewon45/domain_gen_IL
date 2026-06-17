#!/usr/bin/env python3
"""Script for generating commands/jobs."""
import argparse


def write_commands(output_path, commands):
    with open(output_path, "w") as output_file:
        for index, command in enumerate(commands):
            end = "" if index == len(commands) - 1 else "\n"
            output_file.write(command + end)


def build_base_call(args, lr, batch_size, dropout_p):
    return (
        f"python train_sandbox.py "
        f"--data_dir {args.data_dir} "
        f"--output_dir {args.output_dir} "
        f"--exp_name {args.exp_name} "
        f"--lr {lr} "
        f"--batch_size {batch_size} "
        f"--dropout_p {dropout_p}"
    )


def generate_reproduce_commands(args, base_call, seeds):
    commands = []

    erm_pretr_steps = [0, 400]
    erm_total_steps = [400, 600, 1000]
    penalties = [1000, 5000, 10000, 50000, 100000]
    sd_penalties = [1, 5, 10, 50, 100]
    group_dro_etas = [0.001, 0.01, 0.1, 0.5, 1.0]
    eqrm_alphas = [-100, -500, -1000, -5000, -10000]

    algs_1 = ["groupdro", "sd", "iga"]
    algs_steps_1 = [(a, 1000) for a in algs_1]

    algs_2 = ["irm", "vrex", "eqrm"]
    algs_steps_2 = [(a, 600) for a in algs_2]

    algs_steps = algs_steps_1 + algs_steps_2
    algs_settings = [(a, pretr_s, total_s) for (a, total_s) in algs_steps for pretr_s in erm_pretr_steps]

    train_envs = ["default", "gray"]
    erm_settings = [(e, s) for e in train_envs for s in erm_total_steps]

    for seed in seeds:
        for envs, steps in erm_settings:
            commands.append(
                (
                    f"{base_call} "
                    f"--seed {seed} "
                    f"--erm_pretrain_iters 0 "
                    f"--algorithm erm "
                    f"--train_envs {envs} "
                    f"--steps {steps}"
                ).strip()
            )

        for alg, pretr_steps, total_steps in algs_settings:
            alg_base_call = (
                f"{base_call} "
                f"--seed {seed} "
                f"--erm_pretrain_iters {pretr_steps} "
                f"--lr_cos_sched "
                f"--algorithm {alg} "
                f"--steps {total_steps} "
                f"--save_ckpts"
            )
            if alg in ["irm", "vrex", "iga"]:
                alg_settings = [f"--penalty_weight {pen}" for pen in penalties]
            elif alg == "sd":
                alg_settings = [f"--penalty_weight {pen}" for pen in sd_penalties]
            elif alg == "eqrm":
                alg_settings = [f"--alpha {a}" for a in eqrm_alphas]
            elif alg == "groupdro":
                alg_settings = [f"--groupdro_eta {e}" for e in group_dro_etas]
            else:
                raise ValueError(f"Invalid algorithm selected {alg}.")

            for alg_setting in alg_settings:
                commands.append(f"{alg_base_call} {alg_setting}".strip())

    return commands


def generate_domain_stress_commands(args, base_call, seeds):
    commands = []
    algorithms = [
        ("erm", 600, "--erm_pretrain_iters 0"),
        ("irm", 600, "--erm_pretrain_iters 400 --lr_cos_sched --penalty_weight 1000 --save_ckpts"),
        ("groupdro", 1000, "--erm_pretrain_iters 400 --lr_cos_sched --groupdro_eta 0.1 --save_ckpts"),
        ("iro", 600, "--erm_pretrain_iters 400 --lr_cos_sched --save_ckpts"),
        ("inftask", 600, "--erm_pretrain_iters 400 --lr_cos_sched --save_ckpts"),
    ]

    phase1_train_envs = {
        "2": "0.1,0.2",
        "4": "0.01,0.12,0.5,0.99",
        "6": "0.01,0.12,0.0,0.5,0.7,0.99",
        "8": "0.01,0.12,0.0,0.0,0.14,0.5,0.7,0.99",
    }
    phase2_sample_budgets = {
        "small": "2000,2000,2000,2000",
        "medium": "4000,4000,4000,4000",
        "large": "8000,8000,8000,8000",
    }
    phase3_imbalance_budgets = {
        "balanced": "2000,2000,2000,2000",
        "last_domain_heavy_mild": "2000,2000,2000,8000",
        "last_domain_heavy_strong": "2000,2000,2000,12000",
        "first_domain_heavy_mild": "8000,2000,2000,2000",
        "first_domain_heavy_strong": "12000,2000,2000,2000",
    }

    for seed in seeds:
        for phase_name, train_envs in phase1_train_envs.items():
            for algorithm, steps, extra_args in algorithms:
                commands.append(
                    (
                        f"{base_call} "
                        f"--seed {seed} "
                        f"--algorithm {algorithm} "
                        f"--steps {steps} "
                        f"--train_envs {train_envs} "
                        f"{extra_args}"
                    ).strip()
                )

        fixed_train_envs = phase1_train_envs["4"]
        for budget_name, train_env_sizes in phase2_sample_budgets.items():
            for algorithm, steps, extra_args in algorithms:
                commands.append(
                    (
                        f"{base_call} "
                        f"--seed {seed} "
                        f"--algorithm {algorithm} "
                        f"--steps {steps} "
                        f"--train_envs {fixed_train_envs} "
                        f"--train_env_sizes {train_env_sizes} "
                        f"{extra_args}"
                    ).strip()
                )

        for imbalance_name, train_env_sizes in phase3_imbalance_budgets.items():
            for algorithm, steps, extra_args in algorithms:
                commands.append(
                    (
                        f"{base_call} "
                        f"--seed {seed} "
                        f"--algorithm {algorithm} "
                        f"--steps {steps} "
                        f"--train_envs {fixed_train_envs} "
                        f"--train_env_sizes {train_env_sizes} "
                        f"{extra_args}"
                    ).strip()
                )

    return commands

if __name__ == "__main__":
    # Flags
    parser = argparse.ArgumentParser(description='Generate commands for CMNIST experiments.')
    parser.add_argument('--data_dir', type=str, required=True, help="Absolute path to data directory.")
    parser.add_argument('--output_dir', type=str, required=True, help="Absolute path to output directory.")
    parser.add_argument('--exp_name', type=str, default="reproduce")
    args = parser.parse_args()

    # Base settings
    lr = 1e-4
    batch_size = 25000
    dropout_p = 0.2
    seeds = list(range(10))
    base_call = build_base_call(args, lr, batch_size, dropout_p)

    if args.exp_name == "domain_stress":
        commands = generate_domain_stress_commands(args, base_call, seeds)
    else:
        commands = generate_reproduce_commands(args, base_call, seeds)

    output_path = f"job_scripts/{args.exp_name}.txt"
    write_commands(output_path, commands)
    print(f"Total num experiments = {len(commands)}")
