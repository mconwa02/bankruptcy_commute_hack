# Commute Hackaton to Prectict Bankruptcy

**Author: Michelle Conway**

The dataset pf Polish companies and their finacial accounts. **Train** data set which shows which companies did indeed go bankrupt and a **Test** dataset to predict which companies are most likely to go bankrupt.

Building a model based on financial features to predict which of the companies may go bankrupt, by highlighting which financial makers are the strongest indicators of short-term bankruptcy!

Features represent calculations and ratios of sales, assests, liabilites and profits. The final column indicates whether or not the company went bankrupt within the 5 year period.

![Image](shap_plot_summary.PNG)

Looking mainly at bankrupt companies using random forest model and SHAP violin plot above

* Companies with low previous day sales (marked in blue dots) had a high shap value so had a positive effect on the target and more likely to lead to bankrupcty.

* Companies with low assets to liabilities (marked in blue dots) had a high shap values so more likely to become bankrupt.

* Companies with a medium profit per expenses (marked in purple dots) had high shap values so more likely to become bankrupt