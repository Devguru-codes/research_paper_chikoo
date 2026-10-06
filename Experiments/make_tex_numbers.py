"""Write Paper/results_macros.tex and Paper/tab_*.tex from Results/*.json, so that every number in the manuscript
and in the response letter is generated (not typed). Re-run after any experiment."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("RESULTS_DIR", os.path.join(HERE, "..", "Results"))
PAPER = os.environ.get("TEX_OUT", os.path.join(HERE, "..", "Paper"))
SHORT = {"Majority class": "Maj", "PLS-DA": "PLSDA", "Logistic regression": "LogReg", "SVM (RBF)": "SVM",
         "Random forest": "RF", "XGBoost": "XGB", "1D-CNN": "CNN", "1D-ViT": "ViT", "MT-ST": "MTST"}
J = lambda f: json.load(open(os.path.join(RES, f)))
macros = []


def mac(name, val):
    macros.append(f"\\newcommand{{\\{name}}}{{{val}}}")


def p_fmt(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def pc(x):
    return f"{100 * x:.1f}"


def write(name, body):
    open(os.path.join(PAPER, name), "w", encoding="utf-8").write(body)


# ---------------------------------------------------------------- main + stats
M, S = J("main.json"), J("stats.json")
P = M["protocol"]
mac("ResNviews", P["n_views"]); mac("ResNfruits", P["n_fruits"])
tv = P["test_views_per_fold"]
mac("ResTestViews", f"{min(tv)}--{max(tv)}")
rows = []
for name, sh in SHORT.items():
    s, ci, t = M["summary"][name], S["bootstrap_ci_95"][name]["acc"], S["tests_vs_mtst"].get(name)
    mac(f"Acc{sh}", pc(s["acc"][0])); mac(f"AccSD{sh}", pc(s["acc"][1]))
    mac(f"AccLo{sh}", pc(ci[0])); mac(f"AccHi{sh}", pc(ci[1]))
    mac(f"Fone{sh}", pc(s["f1"][0])); mac(f"Kappa{sh}", f"{s['kappa'][0]:.2f}"); mac(f"Bacc{sh}", pc(s["bacc"][0]))
    if t:
        mac(f"Pt{sh}", p_fmt(t["p_t_holm"])); mac(f"Pw{sh}", p_fmt(t["p_wilcoxon_holm"]))
        mac(f"Diff{sh}", f"{100 * t['mean_diff']:+.1f}")
    nm = r"\textbf{MT-ST (proposed)}" if name == "MT-ST" else name
    rows.append(f"{nm} & {pc(s['acc'][0])} $\\pm$ {pc(s['acc'][1])} & [{pc(ci[0])}, {pc(ci[1])}] & "
                f"{pc(s['f1'][0])} $\\pm$ {pc(s['f1'][1])} & {s['kappa'][0]:.2f} & "
                f"{p_fmt(t['p_t_holm']) if t else '--'} & {p_fmt(t['p_wilcoxon_holm']) if t else '--'} \\\\")
write("tab_cls.tex", "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n\\textbf{Model} & \\textbf{Accuracy} & \\textbf{95\\% CI} & "
      "\\textbf{Macro-F1} & $\\boldsymbol{\\kappa}$ & $\\boldsymbol{p}_{t}$ & $\\boldsymbol{p}_{\\mathrm{W}}$ \\\\\n\\midrule\n"
      + "\n".join(rows[:1]) + "\n\\midrule\n" + "\n".join(rows[1:6]) + "\n\\midrule\n" + "\n".join(rows[6:])
      + "\n\\bottomrule\n\\end{tabular}\n")
lr = [p["lr"] for p in M["chosen_params"]["MT-ST"]]
mac("MTSTlrHigh", sum(1 for x in lr if x == 1e-3)); mac("MTSTlrN", len(lr))
for name, sh in [("MT-ST", "MTST"), ("SVM (RBF)", "SVM")]:
    C = M["summary"][name]["confusion_mean"]  # rows/cols: Unripe, Ripe, Overripe
    mac(f"Conf{sh}UU", f"{C[0][0]:.1f}"); mac(f"Conf{sh}RR", f"{C[1][1]:.1f}"); mac(f"Conf{sh}OO", f"{C[2][2]:.1f}")
    mac(f"Conf{sh}OR", f"{C[2][1]:.1f}"); mac(f"Conf{sh}RO", f"{C[1][2]:.1f}")
pmin = min(t["p_t_holm"] for n, t in S["tests_vs_mtst"].items() if n != "Majority class")
pwmin = min(t["p_wilcoxon_holm"] for n, t in S["tests_vs_mtst"].items() if n != "Majority class")
mac("PminTrained", p_fmt(min(pmin, pwmin)))

# ---------------------------------------------------------------- ablation
if os.path.exists(os.path.join(RES, "ablation.json")):
    A = J("ablation.json")
    base = list(A)[0]
    rows = []
    for k, v in A.items():
        s = v["summary"]
        d = 100 * (s["acc"][0] - A[base]["summary"]["acc"][0])
        short = {"Base MT-ST (4 layers, 8 heads, dropout 0.1, sinusoidal PE)": "Base MT-ST",
                 "w/o positional encoding": "w/o pos. encoding", "Patch size 10 (16 tokens)": "10-band tokens (16)",
                 "Patch size 1 (160 tokens)": "1-band tokens (160)"}.get(k, k)
        rows.append(f"{short} & {pc(s['acc'][0])} $\\pm$ {pc(s['acc'][1])} & {pc(s['f1'][0])} $\\pm$ {pc(s['f1'][1])} & "
                    f"{s['kappa'][0]:.2f} & {'--' if k == base else f'{d:+.1f}'} \\\\")
    write("tab_ablation.tex", "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n\\textbf{Variant} & \\textbf{Accuracy (\\%)} & "
          "\\textbf{Macro-F1 (\\%)} & $\\boldsymbol{\\kappa}$ & $\\boldsymbol{\\Delta}$\\textbf{Acc.} \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    mac("AblBaseAcc", pc(A[base]["summary"]["acc"][0]))
    tags = {"w/o positional encoding": "AblNoPE", "w/o dropout": "AblNoDrop", "2 encoder layers": "AblTwoL",
            "4 attention heads": "AblFourH", "16 attention heads": "AblSixteenH",
            "Patch size 10 (16 tokens)": "AblPatchTen", "Patch size 1 (160 tokens)": "AblPatchOne"}
    for k, t in tags.items():
        if k in A:
            mac(t, pc(A[k]["summary"]["acc"][0])); mac(t + "Kappa", f"{A[k]['summary']['kappa'][0]:.2f}")

# ---------------------------------------------------------------- augmentation
if os.path.exists(os.path.join(RES, "augment.json")):
    A = J("augment.json")
    rows = []
    for k, v in A.items():
        s = v["summary"]
        lab = (k.replace("sigma=", "$\\sigma=").replace(", p=", "$, $p=").replace(", no pixel-patch spectra", "$, no patches")
               if "no pixel" in k else k.replace("sigma=", "$\\sigma=").replace(", p=", "$, $p=") + "$")
        rows.append(f"{lab} & {pc(s['acc'][0])} $\\pm$ {pc(s['acc'][1])} & {pc(s['f1'][0])} $\\pm$ {pc(s['f1'][1])} & {s['kappa'][0]:.2f} \\\\")
    write("tab_aug.tex", "\\begin{tabular}{@{}lccc@{}}\n\\toprule\n\\textbf{Augmentation setting} & \\textbf{Accuracy (\\%)} & "
          "\\textbf{Macro-F1 (\\%)} & $\\boldsymbol{\\kappa}$ \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    for k, v in A.items():
        tag = {"sigma=0, p=0.5": "AugZero", "sigma=0.05, p=0.5": "AugFive", "sigma=0.2, p=0.5": "AugTwenty",
               "sigma=0.05, p=1.0": "AugFiveAll", "sigma=0.05, p=0.5, no pixel-patch spectra": "AugNoPatch",
               "sigma=0.1, p=0.5": "AugTen", "sigma=0.01, p=0.5": "AugOne", "sigma=0.025, p=0.5": "AugTwoFive"}.get(k)
        if tag:
            mac(tag, pc(v["summary"]["acc"][0])); mac(tag + "SD", pc(v["summary"]["acc"][1]))

# ---------------------------------------------------------------- attribution
if os.path.exists(os.path.join(RES, "bandsel.json")):
    B = J("bandsel.json")
    st, ag, bs = B["stability"], B["agreement"], B["band_selection_acc"]
    mac("Ktop", B["k_top"])
    for key, sh in [("ig_mean", "IGm"), ("ig_flat", "IGf"), ("att", "Att")]:
        mac(f"Rho{sh}", f"{st[key]['spearman'][0]:.2f}"); mac(f"Jac{sh}", f"{st[key]['jaccard_top_k'][0]:.2f}")
    mac("AgreeIGIG", f"{ag['ig_mean vs ig_flat']:.2f}"); mac("AgreeIGAtt", f"{ag['ig_mean vs att']:.2f}")
    mac("BSall", pc(bs["All 160 bands"][0])); mac("BStopIG", pc(bs["Top-k IG bands"][0]))
    mac("BStopAtt", pc(bs["Top-k attention bands"][0])); mac("BSrand", pc(bs["Random k bands"][0]))
    mac("BSallSD", pc(bs["All 160 bands"][1])); mac("BStopIGSD", pc(bs["Top-k IG bands"][1]))
    mac("BStopAttSD", pc(bs["Top-k attention bands"][1])); mac("BSrandSD", pc(bs["Random k bands"][1]))
    mac("BSp", p_fmt(B["ig_vs_random_wilcoxon_p"]))

# ---------------------------------------------------------------- compute
if os.path.exists(os.path.join(RES, "compute.json")):
    C = J("compute.json")
    rows = []
    for name in ["1D-CNN", "1D-ViT", "MT-ST"]:
        c = C[name]
        L = c["latency"]
        rows.append(f"{name} & {int(c['params']):,} & {c['mflops_per_spectrum']:.1f} & {L['cuda']['median_ms']:.2f} & "
                    f"{L['cpu_1thr']['median_ms']:.2f} & {L['cpu_4thr']['median_ms']:.2f} & {c['sec_per_epoch']:.2f} & "
                    f"{c['peak_vram_mb']:.0f} \\\\".replace(",", "{,}"))
    write("tab_compute.tex", "\\begin{tabular}{@{}lccccccc@{}}\n\\toprule\n\\textbf{Model} & \\textbf{Params} & "
          "\\textbf{MFLOPs} & \\multicolumn{3}{c}{\\textbf{Latency, batch 1 (ms)}} & \\textbf{s/epoch} & \\textbf{Peak VRAM} \\\\\n"
          "\\cmidrule(lr){4-6}\n & & & GPU & CPU, 1 thr. & CPU, 4 thr. & (GPU) & (MB) \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    c = C["MT-ST"]
    gpu = C["hardware"]["gpu"]
    mac("GPUname", (gpu if gpu.startswith("NVIDIA") else "NVIDIA " + gpu).replace("NVIDIA ", "NVIDIA~"))
    cpu = C["hardware"].get("cpu", "laptop CPU")
    for junk in ["(R)", "(TM)", "12th Gen ", " CPU"]:
        cpu = cpu.replace(junk, "")
    cpu = " ".join(cpu.split()).replace(" @ ", ", ").replace("GHz", "\\,GHz")
    mac("CPUname", cpu)
    mac("MTSTgpuMs", f"{c['latency']['cuda']['median_ms']:.2f}"); mac("MTSTcpuOneMs", f"{c['latency']['cpu_1thr']['median_ms']:.2f}")
    mac("MTSTcpuFourMs", f"{c['latency']['cpu_4thr']['median_ms']:.2f}"); mac("MTSTmflops", f"{c['mflops_per_spectrum']:.1f}")
    mac("MTSTvram", f"{c['peak_vram_mb']:.0f}"); mac("MTSTtrainRows", int(c["train_rows"]))

# ---------------------------------------------------------------- pre-registered improvements
if os.path.exists(os.path.join(RES, "improve_stats.json")):
    I, IS = J("improve.json"), J("improve_stats.json")
    ci = IS["bootstrap_ci_95"]
    order = [("SVM (RBF)", I["summary"]["SVM (RBF)"], "SVM (RBF)"),
             ("SVM (RBF), class-balanced", I["summary"]["SVM (RBF), class-balanced"], "SVM (RBF), class-balanced"),
             ("MT-ST (reference, fixed lr)", I["summary"]["MT-ST (reference, fixed lr)"], "MT-ST (reference)"),
             ("MT-ST + class-balanced loss", I["summary"]["MT-ST + class-balanced loss"], "+ class-balanced loss"),
             ("MT-ST + self-supervised pretraining", I["summary"]["MT-ST + self-supervised pretraining"], "+ self-supervised pretraining"),
             ("MT-ST + 5-seed ensemble", I["summary"]["MT-ST + 5-seed ensemble"], "+ 5-seed ensemble"),
             ("MT-ST+ (balanced + pretraining + ensemble)", I["summary"]["MT-ST+ (balanced + pretraining + ensemble)"],
              "\\textbf{MT-ST+ (all three; pre-specified)}")]
    rows = [f"{lab} & {pc(s['acc'][0])} $\\pm$ {pc(s['acc'][1])} & [{pc(ci[k][0])}, {pc(ci[k][1])}] & "
            f"{pc(s['f1'][0])} $\\pm$ {pc(s['f1'][1])} & {s['kappa'][0]:.2f} \\\\" for k, s, lab in order]
    write("tab_improve.tex", "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n\\textbf{Model} & \\textbf{Accuracy (\\%)} & "
          "\\textbf{95\\% CI} & \\textbf{Macro-F1 (\\%)} & $\\boldsymbol{\\kappa}$ \\\\\n\\midrule\n" + "\n".join(rows[:2])
          + "\n\\midrule\n" + "\n".join(rows[2:]) + "\n\\bottomrule\n\\end{tabular}\n")
    tags = {"MT-ST + class-balanced loss": "ImpBal", "MT-ST + self-supervised pretraining": "ImpPre",
            "MT-ST + 5-seed ensemble": "ImpEns", "MT-ST+ (balanced + pretraining + ensemble)": "ImpPlus",
            "SVM (RBF), class-balanced": "ImpSVMbal", "MT-ST (reference, fixed lr)": "ImpRef", "SVM (RBF)": "ImpSVM"}
    for k, t in tags.items():
        s = I["summary"][k]
        mac(t, pc(s["acc"][0])); mac(t + "SD", pc(s["acc"][1])); mac(t + "Fone", pc(s["f1"][0]))
        mac(t + "Kappa", f"{s['kappa'][0]:.2f}"); mac(t + "Lo", pc(ci[k][0])); mac(t + "Hi", pc(ci[k][1]))
    for k, t in [("SVM (RBF)", "VsSVM"), ("SVM (RBF), class-balanced", "VsSVMbal"), ("MT-ST (reference, fixed lr)", "VsMTST")]:
        x = IS["tests_mtstplus_vs"][k]
        mac(t + "Diff", f"{100 * x['mean_diff']:+.1f}"); mac(t + "Pt", p_fmt(x["p_t"])); mac(t + "Pw", p_fmt(x["p_wilcoxon"]))
        mac(t + "Better", x["folds_mtstplus_better"]); mac(t + "Tied", x["folds_tied"]); mac(t + "Worse", x["folds_worse"])
    C = I["summary"]["MT-ST+ (balanced + pretraining + ensemble)"]["confusion_mean"]
    mac("ConfPlusOO", f"{C[2][2]:.1f}"); mac("ConfPlusOR", f"{C[2][1]:.1f}")

# ---------------------------------------------------------------- tabular multi-task experiment (dataset.csv)
if os.path.exists(os.path.join(RES, "csv_main.json")):
    CM = J("csv_main.json")
    CS = J("csv_stats.json") if os.path.exists(os.path.join(RES, "csv_stats.json")) else None
    Sm, Pr = CM["summary"], CM["protocol"]
    mac("CsvRows", Pr["n_rows"]); mac("CsvRecords", Pr["n_records"]); mac("CsvBands", Pr["bands"])
    trp = Pr["test_records_per_fold"]
    mac("CsvTestRecords", f"{min(trp)}--{max(trp)}")
    for t, sh in [("sugar", "Sug"), ("carbohydrate", "Carb")]:
        ts = Pr["target_stats"][t]
        mac(f"Csv{sh}Min", f"{ts['min']:.1f}"); mac(f"Csv{sh}Max", f"{ts['max']:.1f}")
        mac(f"Csv{sh}Mean", f"{ts['mean']:.1f}"); mac(f"Csv{sh}SD", f"{ts['sd']:.1f}")
    REGM = [("PLS regression", "PLS"), ("Ridge regression", "Ridge"), ("SVR (RBF)", "SVR"), ("Random forest", "RFr"),
            ("XGBoost", "XGBr"), ("1D-CNN (multi-task)", "CNNmt"), ("1D-ViT (multi-task)", "ViTmt"),
            ("MT-ST (multi-task, lambda_cls=0.5)", "MTSTmt")]
    rows = []
    for name, sh in REGM:
        s = Sm[name]
        cells = []
        for t, tsh in [("sugar", "Sug"), ("carbohydrate", "Carb")]:
            v = s[t]
            mac(f"Rmse{tsh}{sh}", f"{v['rmse'][0]:.2f}"); mac(f"Rtwo{tsh}{sh}", f"{v['r2'][0]:.2f}")
            mac(f"Mae{tsh}{sh}", f"{v['mae'][0]:.2f}"); mac(f"Rpd{tsh}{sh}", f"{v['rpd'][0]:.2f}")
            cells += [f"{v['rmse'][0]:.2f} $\\pm$ {v['rmse'][1]:.2f}", f"{v['mae'][0]:.2f}", f"{v['r2'][0]:.2f}", f"{v['rpd'][0]:.2f}"]
        lab = {"MT-ST (multi-task, lambda_cls=0.5)": "\\textbf{MT-ST (multi-task)}"}.get(name, name)
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    write("tab_reg.tex", "\\begin{tabular}{@{}lcccccccc@{}}\n\\toprule\n & \\multicolumn{4}{c}{\\textbf{Sugar (\\%)}} & "
          "\\multicolumn{4}{c}{\\textbf{Carbohydrate (\\%)}} \\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n\\textbf{Model} & RMSE & MAE & "
          "$R^2$ & RPD & RMSE & MAE & $R^2$ & RPD \\\\\n\\midrule\n" + "\n".join(rows[:5]) + "\n\\midrule\n" + "\n".join(rows[5:])
          + "\n\\bottomrule\n\\end{tabular}\n")
    best = {t: min([n for n, _ in REGM if n != "MT-ST (multi-task, lambda_cls=0.5)"], key=lambda n: Sm[n][t]["rmse"][0])
            for t in ["sugar", "carbohydrate"]}
    lab_of = dict(REGM)
    mac("CsvBestSug", best["sugar"]); mac("CsvBestCarb", best["carbohydrate"])
    mac("CsvBestSugRmse", f"{Sm[best['sugar']]['sugar']['rmse'][0]:.2f}")
    mac("CsvBestCarbRmse", f"{Sm[best['carbohydrate']]['carbohydrate']['rmse'][0]:.2f}")
    mac("CsvBestSugRtwo", f"{Sm[best['sugar']]['sugar']['r2'][0]:.2f}")
    mac("CsvBestCarbRtwo", f"{Sm[best['carbohydrate']]['carbohydrate']['r2'][0]:.2f}")
    if CS:
        for t, tsh in [("sugar", "Sug"), ("carbohydrate", "Carb")]:
            x = CS["tests_vs_mtst"][t][best[t]]
            mac(f"CsvP{tsh}Best", p_fmt(x["p_t_holm"])); mac(f"CsvPw{tsh}Best", p_fmt(x["p_wilcoxon_holm"]))
            ci = CS["bootstrap_ci_95"]["MT-ST (multi-task, lambda_cls=0.5)"][t]
            mac(f"CsvRtwoLo{tsh}", f"{ci['r2'][0]:.2f}"); mac(f"CsvRtwoHi{tsh}", f"{ci['r2'][1]:.2f}")
            ps = [v["p_t_holm"] for v in CS["tests_vs_mtst"][t].values()]
            mac(f"CsvPmin{tsh}", p_fmt(min(ps)))
            mac(f"CsvRmseLo{tsh}", f"{ci['rmse'][0]:.2f}"); mac(f"CsvRmseHi{tsh}", f"{ci['rmse'][1]:.2f}")
            for name, sh in REGM:
                if name in CS["tests_vs_mtst"][t]:
                    x = CS["tests_vs_mtst"][t][name]
                    mac(f"CsvPt{tsh}{sh}", p_fmt(x["p_t_holm"])); mac(f"CsvPw{tsh}{sh}", p_fmt(x["p_wilcoxon_holm"]))
            cls_names = [n for n, _ in REGM[:5]]
            mac(f"CsvPtMax{tsh}Classical", p_fmt(max(CS["tests_vs_mtst"][t][n]["p_t_holm"] for n in cls_names)))
            mac(f"CsvPwMax{tsh}Classical", p_fmt(max(CS["tests_vs_mtst"][t][n]["p_wilcoxon_holm"] for n in cls_names)))
    # task-weighting study
    LAM = [("MT-ST classification only", "classification only", "ClsOnly"),
           ("MT-ST regression only (lambda_cls=0)", "$\\lambda_{\\mathrm{cls}}=0$ (regr. only)", "LamZero"),
           ("MT-ST lambda_cls=0.25", "$\\lambda_{\\mathrm{cls}}=0.25$", "LamQuarter"),
           ("MT-ST lambda_cls=0.5 (fixed lr)", "$\\lambda_{\\mathrm{cls}}=0.5$ (default)", "LamHalf"),
           ("MT-ST lambda_cls=1", "$\\lambda_{\\mathrm{cls}}=1$", "LamOne"),
           ("MT-ST lambda_cls=2", "$\\lambda_{\\mathrm{cls}}=2$", "LamTwo")]
    rows = []
    for name, lab, sh in LAM:
        s = Sm[name]
        f = lambda t, k: f"{s[t][k][0]:.2f}" if t in s else "--"
        acc = f"{100 * s['cls']['acc'][0]:.1f}" if "cls" in s else "--"
        f1 = f"{100 * s['cls']['f1'][0]:.1f}" if "cls" in s else "--"
        rows.append(f"{lab} & {f('sugar', 'rmse')} & {f('sugar', 'r2')} & {f('carbohydrate', 'rmse')} & "
                    f"{f('carbohydrate', 'r2')} & {acc} & {f1} \\\\")
        if "sugar" in s:
            mac(f"{sh}RmseSug", f("sugar", "rmse")); mac(f"{sh}RtwoSug", f("sugar", "r2"))
            mac(f"{sh}RmseCarb", f("carbohydrate", "rmse")); mac(f"{sh}RtwoCarb", f("carbohydrate", "r2"))
        if "cls" in s:
            mac(f"{sh}Acc", acc)
    write("tab_lambda.tex", "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n & \\multicolumn{2}{c}{\\textbf{Sugar}} & "
          "\\multicolumn{2}{c}{\\textbf{Carbohydrate}} & \\multicolumn{2}{c}{\\textbf{Stage}} \\\\\n"
          "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n\\textbf{Objective} & RMSE & $R^2$ & RMSE & "
          "$R^2$ & Acc. (\\%) & F1 (\\%) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    mac("CsvMTSTAcc", f"{100 * Sm['MT-ST (multi-task, lambda_cls=0.5)']['cls']['acc'][0]:.1f}")
    mac("CsvSVMAcc", f"{100 * Sm['SVM (RBF) classifier']['cls']['acc'][0]:.1f}")
    mac("CsvMajAcc", f"{100 * Sm['Majority class']['cls']['acc'][0]:.1f}")

write("results_macros.tex", "% generated by Experiments/make_tex_numbers.py -- do not edit by hand\n" + "\n".join(macros) + "\n")
print(f"{len(macros)} macros written")
