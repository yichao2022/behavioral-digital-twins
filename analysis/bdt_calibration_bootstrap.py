import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import logit

# 加载数据
df = pd.read_csv('/tmp/behavioral-digital-twins/results/heldout_dce_predictions.csv')

# 只保留 matched observations
df = df[df['matched_llm_state'].notna()].copy()
# 列名已存在：y (y_true), p_dce, p_llm, p_bdt
df.rename(columns={'y': 'y_true'}, inplace=True)

eps = 1e-6
def safe_logit(p):
    p = np.clip(p, eps, 1 - eps)
    return logit(p)

def fit_calibration(data):
    try:
        y = data['y_true'].values
        X = safe_logit(data['p_bdt'].values)
        X = sm.add_constant(X)
        model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        return model.params[0], model.params[1]
    except:
        return np.nan, np.nan

# Respondent-level Cluster Bootstrap
np.random.seed(42)
n_boot = 2000
alphas, betas = [], []
unique_respondents = df['respondent_id'].unique()

for i in range(n_boot):
    sampled_ids = np.random.choice(unique_respondents, size=len(unique_respondents), replace=True)
    boot_df = df[df['respondent_id'].isin(sampled_ids)]
    a, b = fit_calibration(boot_df)
    if not np.isnan(a):
        alphas.append(a)
        betas.append(b)

alphas = np.array(alphas)
betas = np.array(betas)

# Save bootstrap samples
boot_df = pd.DataFrame({'alpha': alphas, 'beta': betas})
boot_df.to_csv('results/bdt_calibration_bootstrap_2000.csv', index=False)

# Summary
print(f"Static-BDT Calibration (B_matched, N={len(df)})")
print(f"Alpha (Intercept): {np.mean(alphas):.3f} [{np.percentile(alphas, 2.5):.3f}, {np.percentile(alphas, 97.5):.3f}]")
print(f"Beta (Slope): {np.mean(betas):.3f} [{np.percentile(betas, 2.5):.3f}, {np.percentile(betas, 97.5):.3f}]")
