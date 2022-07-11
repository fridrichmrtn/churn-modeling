# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # NOTE: MOVE THIS TO PROFIT DEV IPYNB
# MAGIC 
# MAGIC ### Profit estimation
# MAGIC 
# MAGIC 
# MAGIC #### Product margin
# MAGIC 
# MAGIC Unfortunately, margin on user-item transaction are omitted in the available datasets. To include the profit aspect of the customer relationship management, we propose simulation approach describing changes in product margin over time, which are consequently used for estimating customer profit. That is, for each of the product, we draw from initial random normal distribution. This draw serves as a starting point for the random walk, which we simulate using draws from random normal distribution for step difference and put them together using cumulative sum. In other words, we use one dimensional random walk with random normal steps. For product *p* in time *t*, we estimate the margin *m* like this:
# MAGIC    
# MAGIC $$m^{p}_t = Normal(\mu_0, \sigma_0)+\sum^{t}_{n=1}{Normal(\mu_{diff}, \sigma_{diff})}$$
# MAGIC 
# MAGIC where the first element represents the starting draw, and the second element represents the cumulative sum of difference draws. For simplicity, we assume that initial variability across products and variability of the product, within the observed period, are the same. Thus, we set \\(\mu_{diff} = 0\\), and \\(\sigma_{diff}=\frac{\sigma_0}{\sqrt{t}}\\).  As a result, we are able to estimate product profit with respect to just parameters of the initial random draw \\(\mu_{0}\\) and \\(\sigma_{0}\\).
# MAGIC 
# MAGIC #### Customer profit
# MAGIC 
# MAGIC The customer profit is computed using product revenue and simulated margin, result is scaled to reflect the target window size. This approach allows us to differentiate customer profit on individual level, indicate changes in customer behavior (partial churn), and can be used as secondary target for modeling. On the other hand, it is fairly limited by the window size and does not reflect on lifecycle length such as CLV. Let as have a customer \\(i\\) at time \\(t\\), the average customer profit can be computed as
# MAGIC 
# MAGIC $$ ACP^{i}_t = \frac{n_t}{t}\sum^{t}_{n=1}{m^{p}_n  r^{p}_n} $$
# MAGIC 
# MAGIC where \\(n_t\\) represent the length of the target window, \\(m^{p}_n\\) stands for simulated margin for product \\(p\\) in the time \\(n\\), and \\(r^{p}_n\\) is the revenue from transaction realized on product \\(p\\) in the time \\(n\\).
# MAGIC 
# MAGIC NOTE: INCLUDE TIME AXIS, REFERENCE OTHER APPROACHES
# MAGIC 
# MAGIC #### Plotting
# MAGIC 
# MAGIC 
# MAGIC * AVGCP VS TIME
# MAGIC * AVGCP DISTRIBUTION
# MAGIC * vs BOTH DATASETS
# MAGIC 
# MAGIC #### Sensitivity analysis
# MAGIC 
# MAGIC  * AVGCP VS mu0, sigma
# MAGIC  * vs BOTH DATASETS
# MAGIC  
# MAGIC 
# MAGIC ### Expected campaign profit
# MAGIC 
# MAGIC The classical approach to retention management is to identify the customers at high risk of churning and persuade them to stay active with promotional offers. Predictive solutions are, however, primarily evaluated in terms of the classification task, not in the retention management context. To reflect on those additional steps, we propose an evaluation procedure inspired by the retention campaign (Neslin et al., 2006) and generated profit frameworks (Tamaddoni et al., 2015).
# MAGIC 
# MAGIC However, we focus on differentiated customer value and 
# MAGIC 
# MAGIC 
# MAGIC $$ \pi_i = p_i[\gamma_i V_i (1-\delta_i)] + (1-p_i)[-\psi_i \delta_i]$$
# MAGIC 
# MAGIC ### Maximum expected campaign profit
