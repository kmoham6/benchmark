import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the combined CSV file into a DataFrame
data = pd.read_csv('result.csv')  # Ensure this matches your actual file name

# Print the column names to check for 'benchmark'
print("Columns in the DataFrame:", data.columns.tolist())

# Ensure the 's' column is numeric
data['s'] = pd.to_numeric(data['s'], errors='coerce')

# Drop rows where 's' could not be converted to numeric
data.dropna(subset=['s'], inplace=True)

# Create a new figure for plotting
fig, ax = plt.subplots()

# Prepare a list to hold the exponents for x-ticks later
exponents = []

# # Define colors for each benchmark
color_map = {
    'core=2': 'blue',
    'core=8': 'green',
    'core=16' :'orange',
    'core=32': 'purple',
    'core=64': 'black',
    'acc': 'red',

}

# Loop through each unique benchmark case
for benchmark in data['benchmark'].unique():
    benchmark_data = data[data['benchmark'] == benchmark].copy()  # Use .copy() to avoid SettingWithCopyWarning

    # Calculate the exponent of 2 for each s value
    benchmark_data['exponent'] = np.log2(benchmark_data['s'])

    # Store the exponents for x-ticks
    exponents.extend(benchmark_data['exponent'].tolist())

   # Set line style and thickness based on the benchmark
    # if benchmark == 'acc':  # Highlight the 'acc' line
    #     linewidth = 10.0  # Make the 'acc' line very thick
    #     linestyle = '-'  # Keep the 'acc' line solid
    #     alpha = 1.0  # Fully opaque
    # else:
    #     linewidth = 2.0  # Thinner lines for others
    #     linestyle = '--'  # Use dashed lines for others
    #     alpha = 0.5  # Make other lines semi-transparent
    
    # Plotting the line chart for each benchmark case with assigned color
    ax.plot(benchmark_data['exponent'], benchmark_data['speedUp'], 
            label=benchmark, color=color_map.get(benchmark, 'black'))  # Default to black if benchmark not found
    # ax.plot(benchmark_data['exponent'], benchmark_data['speedUp'], label=benchmark)

# Set x-ticks to show each exponent value
unique_exponents = np.unique(exponents)  # Get unique exponents for x-ticks
ax.set_xticks(unique_exponents)
ax.set_xticklabels(unique_exponents.astype(int)) 

# # Set the y-ticks to a specified interval, e.g., every 1 unit
# y_tick_interval = 1  # Change this value to your desired interval
# y_min, y_max = 0, 40
# ax.set_ylim(y_min, y_max)
# ax.set_yticks(np.arange(y_min, y_max + y_tick_interval, y_tick_interval))

# Ensure the y-axis starts at 0
y_tick_interval = 2  # Change this value to your desired interval
y_min, y_max = 0, ax.get_ylim()[1]  # Set y_min to 0 and keep the original y_max
ax.set_ylim(y_min, y_max)  # Apply the y_min and y_max
ax.set_yticks(np.arange(y_min, y_max + y_tick_interval, y_tick_interval))

ax.set_xlabel('Number of elements in input sequence (Log2($N_E$))')
ax.set_ylabel('SpeedUp')
ax.set_title('Speed up for different Core Counts vs "acc" Executor')

# Add a legend to identify each benchmark
ax.legend(title='Benchmark', loc='best')

# Save the figure as a JPG file
plt.savefig('line_chart_multiple_cases_compute_risc5_new.jpg', format='jpg')

# Close the plot to avoid displaying it
plt.close()