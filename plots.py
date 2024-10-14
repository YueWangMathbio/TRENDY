import matplotlib.pyplot as plt
import numpy as np

# hESC
groups = ['WENDY', 'TRENDY', 'GENIE3', 'GENIE3-rev',
          'SINCERITIES', 'SINCERITIES-rev', 'NonlinearODEs', 'NonlinearODEs-rev']
values = np.array([[0.4997, 0.0392],
                   [0.5311, 0.0376],
                   [0.5913, 0.0468],
                   [0.6008, 0.0435],
                   [0.4198, 0.0261],
                   [0.4871, 0.0294],
                   [0.5971, 0.0534],
                   [0.6233, 0.0641]])

# Splitting the values into AUROC and AUPRC
auroc_values = values[:, 0]  # AUROC values for each group
auprc_values = values[:, 1]  # AUPRC values for each group

# Number of groups
num_groups = len(groups)

# Create figure and axis
fig, ax = plt.subplots()

# Set bar width and positions
bar_width = 0.35
index = np.arange(num_groups)

# Plot the bars for AUROC and AUPRC
bar1 = ax.bar(index, auroc_values, bar_width, label='AUROC')
bar2 = ax.bar(index + bar_width, auprc_values, bar_width, label='AUPRC')

# Omit the x label and add a y label
ax.set_ylabel('AUC value')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(groups, rotation=45, ha="right")
ax.set_ylim([0, 1])

# Add a legend
ax.legend()

# Display the plot
plt.tight_layout()

plt.savefig('hESC.pdf') 
plt.show()
plt.close()

# THP-1
groups = ['WENDY', 'TRENDY', 'GENIE3', 'GENIE3-rev',
          'SINCERITIES', 'SINCERITIES-rev', 'NonlinearODEs', 'NonlinearODEs-rev']
values = np.array([[0.5261, 0.3972],
 [0.5557, 0.3669],
 [0.4484, 0.3546],
 [0.5506, 0.3781],
 [0.6261, 0.3852],
 [0.5251, 0.3412],
 [0.5338, 0.3486],
 [0.4808, 0.3302]])

# Splitting the values into AUROC and AUPRC
auroc_values = values[:, 0]  # AUROC values for each group
auprc_values = values[:, 1]  # AUPRC values for each group

# Number of groups
num_groups = len(groups)

# Create figure and axis
fig, ax = plt.subplots()

# Set bar width and positions
bar_width = 0.35
index = np.arange(num_groups)

# Plot the bars for AUROC and AUPRC
bar1 = ax.bar(index, auroc_values, bar_width, label='AUROC')
bar2 = ax.bar(index + bar_width, auprc_values, bar_width, label='AUPRC')

# Omit the x label and add a y label
ax.set_ylabel('AUC value')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(groups, rotation=45, ha="right")
ax.set_ylim([0, 1])

# Add a legend
ax.legend()

# Display the plot
plt.tight_layout()
plt.savefig('THP-1.pdf') 
plt.show()

# DREAM4
groups = ['WENDY', 'TRENDY', 'GENIE3', 'GENIE3-rev',
          'SINCERITIES', 'SINCERITIES-rev', 'NonlinearODEs', 'NonlinearODEs-rev']
values = np.array([[0.4899, 0.2080],
 [0.5341, 0.2177],
 [0.5636, 0.2286],
 [0.4589, 0.1799],
 [0.4908, 0.1919],
 [0.4995, 0.2034],
 [0.4806, 0.1705],
 [0.5712, 0.2452]])

# Splitting the values into AUROC and AUPRC
auroc_values = values[:, 0]  # AUROC values for each group
auprc_values = values[:, 1]  # AUPRC values for each group

# Number of groups
num_groups = len(groups)

# Create figure and axis
fig, ax = plt.subplots()

# Set bar width and positions
bar_width = 0.35
index = np.arange(num_groups)

# Plot the bars for AUROC and AUPRC
bar1 = ax.bar(index, auroc_values, bar_width, label='AUROC')
bar2 = ax.bar(index + bar_width, auprc_values, bar_width, label='AUPRC')

# Omit the x label and add a y label
ax.set_ylabel('AUC value')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(groups, rotation=45, ha="right")
ax.set_ylim([0, 1])

# Add a legend
ax.legend()

# Display the plot
plt.tight_layout()
plt.savefig('DREAM4.pdf') 
plt.show()


# X-axis values (time)
x_values = np.linspace(0.1, 1.0, 10)  # 10 ticks from 0.1 to 1.0

# Y-axis values for the lines
wendy_values = [0.7819, 0.7025, 0.6658, 0.6567, 0.6517, 0.6493, 0.6434, 0.6378, 0.6356, 0.6298]
trendy_values = [0.9254, 0.8900, 0.8670, 0.8620, 0.8527, 0.8422, 0.8354, 0.8305, 0.8246, 0.8177]
genie3_values = [0.4384, 0.4374, 0.4427, 0.4403, 0.4264, 0.4079, 0.3858, 0.3704, 0.3588, 0.3536]
genie3_rev_values = [0.9382, 0.9120, 0.8880, 0.8770, 0.8661, 0.8551, 0.8471, 0.8441, 0.8407, 0.8346]

# Y-values for horizontal lines
sincerities_y = 0.6783
sincerities_rev_y = 0.7964
nonlinearodes_y = 0.5076
nonlinearodes_rev_y = 0.5976

# Create the plot
plt.figure(figsize=(9, 5))

# Plot the line for WENDY (dashed)
plt.plot(x_values, wendy_values, label='WENDY', linestyle='--', marker='o', color='blue')

# Plot the line for TRENDY (solid)
plt.plot(x_values, trendy_values, label='TRENDY', linestyle='-', marker='o', color='blue')

# Plot the line for GENIE3 (dashed)
plt.plot(x_values, genie3_values, label='GENIE3', linestyle='--', marker='o', color='red')

# Plot the line for GENIE3-rev (solid)
plt.plot(x_values, genie3_rev_values, label='GENIE3-rev', linestyle='-', marker='o', color='red')

# Add the horizontal lines
plt.hlines(sincerities_y, x_values[0], x_values[-1], colors='orange', linestyle='--', label='SINCERITIES')
plt.hlines(sincerities_rev_y, x_values[0], x_values[-1], colors='orange', linestyle='-', label='SINCERITIES-rev')
plt.hlines(nonlinearodes_y, x_values[0], x_values[-1], colors='green', linestyle='--', label='NonlinearODEs')
plt.hlines(nonlinearodes_rev_y, x_values[0], x_values[-1], colors='green', linestyle='-', label='NonlinearODEs-rev')

# Set x-axis label and ticks
plt.xlabel('time', fontsize=14)
plt.xticks(np.linspace(0.1, 1.0, 10))  # X-axis ticks at 0.1 intervals

# Set y-axis label and range
plt.ylabel('AUROC value', fontsize=14)
plt.ylim(0, 1)

# Add the legend outside the plot area
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust plot to make room for legend

plt.savefig('SINC_AUROC.pdf') 
# Show the plot
plt.show()





# X-axis values (time)
x_values = np.linspace(0.1, 1.0, 10)  # 10 ticks from 0.1 to 1.0

# Y-axis values for the lines
wendy_values = [0.7014, 0.6433, 0.6191, 0.6086, 0.6016, 0.5960, 0.5904, 0.5860, 0.5829, 0.5801]     
trendy_values = [0.8803, 0.8260, 0.7879, 0.7690, 0.7449, 0.7234, 0.7086, 0.6990, 0.6930, 0.6870]
genie3_values = [0.5538, 0.5387, 0.5281, 0.5180, 0.5086, 0.4979, 0.4879, 0.4800, 0.4757, 0.4728]
genie3_rev_values = [0.8984, 0.8512, 0.8107, 0.7853, 0.7606, 0.7368, 0.7188, 0.7094, 0.7027, 0.6982]

# Y-values for horizontal lines
sincerities_y = 0.5829
sincerities_rev_y = 0.6637
nonlinearodes_y = 0.5313
nonlinearodes_rev_y = 0.5658

# Create the plot
plt.figure(figsize=(9, 5))

# Plot the line for WENDY (dashed)
plt.plot(x_values, wendy_values, label='WENDY', linestyle='--', marker='o', color='blue')

# Plot the line for TRENDY (solid)
plt.plot(x_values, trendy_values, label='TRENDY', linestyle='-', marker='o', color='blue')

# Plot the line for GENIE3 (dashed)
plt.plot(x_values, genie3_values, label='GENIE3', linestyle='--', marker='o', color='red')

# Plot the line for GENIE3-rev (solid)
plt.plot(x_values, genie3_rev_values, label='GENIE3-rev', linestyle='-', marker='o', color='red')

# Add the horizontal lines
plt.hlines(sincerities_y, x_values[0], x_values[-1], colors='orange', linestyle='--', label='SINCERITIES')
plt.hlines(sincerities_rev_y, x_values[0], x_values[-1], colors='orange', linestyle='-', label='SINCERITIES-rev')
plt.hlines(nonlinearodes_y, x_values[0], x_values[-1], colors='green', linestyle='--', label='NonlinearODEs')
plt.hlines(nonlinearodes_rev_y, x_values[0], x_values[-1], colors='green', linestyle='-', label='NonlinearODEs-rev')

# Set x-axis label and ticks
plt.xlabel('time', fontsize=14)
plt.xticks(np.linspace(0.1, 1.0, 10))  # X-axis ticks at 0.1 intervals

# Set y-axis label and range
plt.ylabel('AUPRC value', fontsize=14)
plt.ylim(0, 1)

# Add the legend outside the plot area
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust plot to make room for legend

plt.savefig('SINC_AUPRC.pdf') 
# Show the plot
plt.show()

