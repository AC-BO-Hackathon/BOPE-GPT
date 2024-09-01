import plotly.graph_objects as go
import numpy as np

# Assuming necessary imports and data setup is done prior to this function definition

def ci(y):
    return 1.96 * y.std(axis=0) / np.sqrt(y.shape[0])

def plot_and_save_html(NUM_BATCHES, algo, best_vals, q_eubo, q_comp, q_data):
    algo_labels = {
        "rand": "Random Exploration",
        "EUBO-LLM": "EUBO-LLM",
        "EUBO": "EUBO", 
    }
    
    iters = list(range(NUM_BATCHES + 1))
    fig = go.Figure()
    
        ys = np.vstack(best_vals[algo])
    fig.add_trace(go.Scatter(x=iters, y=ys.mean(axis=0), mode='lines+markers',
                                name=algo_labels[algo],
                                error_y=dict(type='data', array=ci(ys), visible=True)))
    
    fig.update_layout(
        title="Obj 1: maximising every output at the same time, with the same importance",
        xaxis_title=f"Number of queries q = {q_eubo}, num_initial_comp = {q_comp}, num_initial_samp = {q_data}",
        xaxis=dict(tickmode='linear', tick0=0, dtick=1),
        yaxis_title="Best observed value\nas evaluated in the synthetic utility function",
        font=dict(size=14),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    # Save the plot as an HTML file
    fig.write_html("comparison_first_case_1_corrected.html")

# Example usage of the function
# plot_and_save_html(NUM_BATCHES, algos, best_vals, q_eubo, q_comp, q_data)
