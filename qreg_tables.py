import os
import numpy as np
import pandas as pd
import scipy.stats as scs

import statsmodels.formula.api as smf
import statsmodels.api as sm

from functools import partial

max_iter = 350000

fpath_out = "/work/users/mtorrassa/biofire-idh/data_new/"
quantiles = [.85, .90 ,.95]

def use_f_2(x, num_decimals):
    return f"%.{num_decimals}f" % x

# the number of columns can be passed to this function
use_f = lambda x: partial(use_f_2, num_decimals=x)

def wald_test(model):

# Get the coefficients (excluding intercept)
    beta = model.params[1:]  # Coefficients for X and X_squared

    # Get the covariance matrix of the estimated coefficients
    cov_matrix = model.cov_params().iloc[1:, 1:]  # Covariance for X and X_squared

    # Wald statistic calculation: W = beta' * (cov^-1) * beta
    wald_stat = np.dot(np.dot(beta.T, np.linalg.inv(cov_matrix)), beta)

    # Degrees of freedom is the number of coefficients (excluding the intercept)
    df = len(beta)

    # Compute p-value based on chi-squared distribution
    p_value = scs.chi2.sf(wald_stat, df)

    # print(f"Wald Test Statistic: {wald_stat:.4f}")
    # print(f"p-value: {p_value:.4f}")

    return wald_stat, p_value

def aic_test(Y, X, Y_pr, model):
    # return aic metric
    k_params = model.df_model + model.k_constant
    
    # Estimate maximized log likelihood
    nobs = float(X.shape[0])
    nobs2 = nobs / 2.0
    nobs = float(nobs)
    resid = Y - Y_pr
    ssr = np.sum((resid)**2)
    llf = -nobs2*np.log(2*np.pi) - nobs2*np.log(ssr / nobs) - nobs2

    return -2 * llf + 2 * k_params


def best_poly_quantile_regression(Y, X, quantile=0.5, max_degree=5, verbose=True):
    best_degree = None
    best_aic = np.inf  # Initialize to a large value
    best_model = None

    # Data preparation
    data = pd.DataFrame({'X': X, 'Y': Y})

    # Iterate over polynomial degrees from 1 to max_degree
    for degree in range(1, max_degree + 1):
        # Construct formula for polynomial regression up to the current degree
        formula = 'Y ~ ' + ' + '.join([f'np.power(X, {i})' for i in range(1, degree + 1)])

        # Fit quantile regression for the current degree
        model = smf.quantreg(formula, data).fit(q=quantile, max_iter=max_iter)
        
        # Predict the values for Y
        Y_pr = model.predict(data)
        
        # Calculate AIC for the current model
        aic_value = aic_test(data['Y'], data['X'], Y_pr, model)
        
        # print(f'Degree {degree}: AIC = {aic_value:.4f}')

        w_stat, w_pval = wald_test(model)
        # keep only the statistically significant regression
        if w_pval < 0.05:
            # Update the best model if current AIC is smaller
            if aic_value < best_aic:
                best_aic = aic_value
                best_degree = degree
                best_model = model

    if verbose:
        print(f'Best Polynomial Degree: {best_degree} with AIC = {best_aic:.4f}\n')
    return best_degree, best_model, best_aic



df2 = pd.read_csv(os.path.join(fpath_out, 'coms-fire-bioindex-fd.csv'))
df2['frt']=np.log10(df2['frt'])

with open("qreg_tables.txt", "w") as tab_file:

    tab_file.write("# COMPOSITIONAL DIVERSITY\n")
    for n in [10,50]:
        for biome in ['med', 'bor']:
            for bioind in ["srichness", "isimpson"]:

                df_qreg = pd.DataFrame(index=['poly order', 'quantile', 'AIC', 'R-squared', 'Wald test', 'p-value']).T

                data = df2[(df2['N'] == n) & (df2['biome'] == biome)]
                data = data.rename(columns={'frt':'X', bioind:'Y'})
                if not data.empty:

                    data['X_2']=data['X']**2
                    data['X_3']=data['X']**3
                    data['X_4']=data['X']**4
                    data['X_5']=data['X']**5

                    for q in quantiles:
                        for poly in range(1,6):

                            if poly==1:
                                model = smf.quantreg('Y ~ I(X)', data)
                                X = data['X']
                            elif poly==2:
                                model = smf.quantreg('Y ~ I(X) + I(X**2)', data)
                                X = data[['X', 'X_2']]
                            elif poly==3:
                                model = smf.quantreg('Y ~ I(X) + I(X**2) + I(X**3)', data)
                                X = data[['X', 'X_2', 'X_3']]
                            elif poly==4:
                                model = smf.quantreg('Y ~ I(X) + I(X**2) + I(X**3) + I(X**4)', data)
                                X = data[['X', 'X_2', 'X_3', 'X_4']]
                            elif poly==5:
                                model = smf.quantreg('Y ~ I(X) + I(X**2) + I(X**3) + I(X**4) + I(X**5)', data)
                                X = data[['X', 'X_2', 'X_3', 'X_4', 'X_5']]
                            else:
                                print('Error')

                            X = sm.add_constant(X)
                            lr = model.fit(q=q)
                            pr = lr.predict(X)

                            AIC = aic_test(data['Y'], data['X'], pr, model)
                            Rsq = lr.prsquared
                            w_stat, w_pval = wald_test(lr)
                            
                            df_temp = pd.DataFrame([int(poly), round(q,2), round(AIC,4), round(Rsq,4), round(w_stat,4), round(w_pval,4)], index=['poly order', 'quantile', 'AIC', 'R-squared', 'Wald test', 'p-value'])
                            df_qreg = pd.concat([df_qreg, df_temp.T], axis=0)

                    caption=f'Quantile regression statistics of the {bioind} and fire return time relationship for the $N={n}$ {biome} simulations, across quantiles ($75th$, $85t$ and $95th$) and polynomial orders (1 to 5). The table includes the Akaike Information Criterion (AIC), pseudo R-squared, Wald test statistic, and p-value. (*) indicates the best polynomial order for each quantile.'
                    df_qreg = df_qreg.set_index("quantile")

                    # caption=f"Quantile Regression Statistics of the {bioind} - fire return time relationship for the {biome} N={n} communities. \
                    #         statistics are reported for different quantiles (75, 85 and 95th) and polynomial orders (1, 2, 3, 4 and 5): \
                    #         Akaike Information Criterion (AIC), pseudo R-sqaured, Wald test statistic and p-value."

                    latex_table = df_qreg.reset_index().to_latex(
                        index=False,
                        formatters = [use_f(2), use_f(0), use_f(1), use_f(3), use_f(1), use_f(1)],
                        # float_format="%.3f",
                        caption=caption,
                        label=f"tab:qreg_{bioind[:5]}_{biome}{n}",
                        escape=False,
                        column_format="ll|cccc",
                        position='h!',
                        multirow = True
                    )

                    # Post-process the LaTeX table to match the desired format
                    latex_table = latex_table.replace("\\bottomrule\n", "")

                    # Add additional custom rows and multirow format
                    latex_table = latex_table.replace(
                        "\\toprule",
                        "\\multirow{2}{*}{\\textbf{Quantile}} & \\textbf{Polynomial} & \\multirow{2}{*}{\\textbf{AIC}} & \\multirow{2}{*}{\\textbf{R-squared}} & \\multicolumn{2}{c}{\\textbf{Wald test}} \\"
                    ).replace(
                        "quantile & poly order & AIC & R-squared & Wald test & p-value \\",
                        "& \\textbf{Order} &  &  & \\textbf{Statistics} & \\textbf{p-value} \\")

                    # Add additional custom rows and multirow format
                    latex_table = latex_table.replace("0.75 & 1","\\multirow{5}{*}{0.75} & 1").replace("0.75 & 2"," & 2").replace("0.75 & 3"," & 3").replace("0.75 & 4"," & 4").replace("0.75 & 5"," & 5"
                                                    ).replace("0.85 & 1","\\midrule\n\\multirow{5}{*}{0.85} & 1").replace("0.85 & 2"," & 2").replace("0.85 & 3"," & 3").replace("0.85 & 4"," & 4").replace("0.85 & 5"," & 5"
                                                    ).replace("0.90 & 1","\\midrule\n\\multirow{5}{*}{0.90} & 1").replace("0.90 & 2"," & 2").replace("0.90 & 3"," & 3").replace("0.90 & 4"," & 4").replace("0.90 & 5"," & 5"
                                                    ).replace("0.95 & 1","\\midrule\n\\multirow{5}{*}{0.95} & 1").replace("0.95 & 2"," & 2").replace("0.95 & 3"," & 3").replace("0.95 & 4"," & 4").replace("0.95 & 5"," & 5")


                    latex_table = latex_table.replace("$75th$, $85t$ and $95th$","$75^{th}$, $85^{th}$ and $95^{th}$")

                    # with open("myfile.txt", "a") as tab_file:
                    tab_file.write(f'{biome} - {n} - {bioind} \n\n')
                    tab_file.write(latex_table)
                    tab_file.write('\n\n')


    tab_file.write("# FUNCTIONAL DIVERSITY\n")
    for n in [10,50]:
        for biome in ['med', 'bor']:

            for bioind in ["frichness", "fdivergence"]:

                df_qreg = pd.DataFrame(index=['poly order', 'quantile', 'AIC', 'R-squared', 'Wald test', 'p-value']).T

                data = df2[(df2['N'] == n) & (df2['biome'] == biome)]
                data = data.rename(columns={'frt':'X', bioind:'Y'})
                if not data.empty:

                    data['X_2']=data['X']**2
                    data['X_3']=data['X']**3
                    data['X_4']=data['X']**4
                    data['X_5']=data['X']**5

                    for q in quantiles:
                        for poly in range(1,6):

                            if poly==1:
                                model = smf.quantreg('Y ~ I(X)', data)
                                X = data['X']
                            elif poly==2:
                                model = smf.quantreg('Y ~ I(X) + I(X**2)', data)
                                X = data[['X', 'X_2']]
                            elif poly==3:
                                model = smf.quantreg('Y ~ I(X) + I(X**2) + I(X**3)', data)
                                X = data[['X', 'X_2', 'X_3']]
                            elif poly==4:
                                model = smf.quantreg('Y ~ I(X) + I(X**2) + I(X**3) + I(X**4)', data)
                                X = data[['X', 'X_2', 'X_3', 'X_4']]
                            elif poly==5:
                                model = smf.quantreg('Y ~ I(X) + I(X**2) + I(X**3) + I(X**4) + I(X**5)', data)
                                X = data[['X', 'X_2', 'X_3', 'X_4', 'X_5']]
                            else:
                                print('Error')

                            X = sm.add_constant(X)
                            lr = model.fit(q=q)
                            pr = lr.predict(X)

                            AIC = aic_test(data['Y'], data['X'], pr, model)
                            Rsq = lr.prsquared
                            w_stat, w_pval = wald_test(lr)
                            
                            df_temp = pd.DataFrame([int(poly), round(q,2), round(AIC,4), round(Rsq,4), round(w_stat,4), round(w_pval,4)], index=['poly order', 'quantile', 'AIC', 'R-squared', 'Wald test', 'p-value'])
                            df_qreg = pd.concat([df_qreg, df_temp.T], axis=0)

                    caption=f'Quantile regression statistics of the {bioind} and fire return time relationship for the $N={n}$ {biome} simulations, across quantiles ($75th$, $85t$ and $95th$) and polynomial orders (1 to 5). The table includes the Akaike Information Criterion (AIC), pseudo R-squared, Wald test statistic, and p-value. (*) indicates the best polynomial order for each quantile.'
                    df_qreg = df_qreg.set_index("quantile")

                    # caption=f"Quantile Regression Statistics of the {bioind} - fire return time relationship for the {biome} N={n} communities. \
                    #         statistics are reported for different quantiles (75, 85 and 95th) and polynomial orders (1, 2, 3, 4 and 5): \
                    #         Akaike Information Criterion (AIC), pseudo R-sqaured, Wald test statistic and p-value."

                    latex_table = df_qreg.reset_index().to_latex(
                        index=False,
                        formatters = [use_f(2), use_f(0), use_f(1), use_f(3), use_f(1), use_f(1)],
                        # float_format="%.3f",
                        caption=caption,
                        label=f"tab:qreg_{bioind[:5]}_{biome}{n}",
                        escape=False,
                        column_format="ll|cccc",
                        position='h!',
                        multirow = True
                    )

                    # Post-process the LaTeX table to match the desired format
                    latex_table = latex_table.replace("\\bottomrule\n", "")

                    # Add additional custom rows and multirow format
                    latex_table = latex_table.replace(
                        "\\toprule",
                        "\\multirow{2}{*}{\\textbf{Quantile}} & \\textbf{Polynomial} & \\multirow{2}{*}{\\textbf{AIC}} & \\multirow{2}{*}{\\textbf{R-squared}} & \\multicolumn{2}{c}{\\textbf{Wald test}} \\"
                    ).replace(
                        "quantile & poly order & AIC & R-squared & Wald test & p-value \\",
                        "& \\textbf{Order} &  &  & \\textbf{Statistics} & \\textbf{p-value} \\")

                    # Add additional custom rows and multirow format
                    latex_table = latex_table.replace("0.75 & 1","\\multirow{5}{*}{0.75} & 1").replace("0.75 & 2"," & 2").replace("0.75 & 3"," & 3").replace("0.75 & 4"," & 4").replace("0.75 & 5"," & 5"
                                                    ).replace("0.85 & 1","\\midrule\n\\multirow{5}{*}{0.85} & 1").replace("0.85 & 2"," & 2").replace("0.85 & 3"," & 3").replace("0.85 & 4"," & 4").replace("0.85 & 5"," & 5"
                                                    ).replace("0.90 & 1","\\midrule\n\\multirow{5}{*}{0.90} & 1").replace("0.90 & 2"," & 2").replace("0.90 & 3"," & 3").replace("0.90 & 4"," & 4").replace("0.90 & 5"," & 5"
                                                    ).replace("0.95 & 1","\\midrule\n\\multirow{5}{*}{0.95} & 1").replace("0.95 & 2"," & 2").replace("0.95 & 3"," & 3").replace("0.95 & 4"," & 4").replace("0.95 & 5"," & 5")

                    latex_table = latex_table.replace("$75th$, $85t$ and $95th$","$75^{th}$, $85^{th}$ and $95^{th}$"
                                                    ).replace("med", "Mediterranean"
                                                    ).replace("bor", "Boreal")

                    # with open("myfile.txt", "a") as tab_file:
                    tab_file.write(f'{biome} - {n} - {bioind} \n\n')
                    tab_file.write(latex_table)
                    tab_file.write('\n\n')