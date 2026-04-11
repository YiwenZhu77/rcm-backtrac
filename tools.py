# tools.py

import libs
import params

def display_timer(start_time, stop_event, label="Elapsed time"):
    """
    Display a dynamic timer with a custom label in hh:mm:ss format.

    Parameters:
    - start_time: The start time of the timer (from time.time()).
    - stop_event: A threading.Event object to stop the timer.
    - label: A string label to display before the time (default: "Elapsed time").
    """
    while not stop_event.is_set():
        elapsed_time = libs.time.time() - start_time
        # Convert to hh:mm:ss
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02.2f}"
        libs.sys.stdout.write(f"\r{label}: {time_str}")
        libs.sys.stdout.flush()
        libs.time.sleep(1)
    elapsed_time = libs.time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02.2f}"
    libs.sys.stdout.write(f"\r{label}: {time_str}\n")
    libs.sys.stdout.flush()
    

def write_progress_index(start_time, stop_event, filename= params.code_path + "bt.index",
                        num_chunks=None, initial_particles=None, added_particles=None,
                        num_cpu=None, i_max=None, j_max=None):

    # Create parent directory if it doesn't exist
    libs.os.makedirs(libs.os.path.dirname(filename), exist_ok=True)
    
    # Overwrite file with header at the start of each run
    with open(filename, 'w') as f:
        current_time = libs.datetime.datetime.now()
        f.write(f"BEGINNING NEW RUN  TODAY IS {current_time.strftime('%Y/%m/%d')}  TIME: {current_time.strftime('%H:%M:%S')}\n")
        f.write("time        #of_chunks  #of_par(initial)  #of_par(added)  #of_cpu  (i_max,j_max)\n")
    
    # Calculate initial delay to align with minute boundaries
    elapsed_time = libs.time.time() - start_time
    initial_delay = 60 - (elapsed_time % 60)  # Seconds until next minute
    if initial_delay < 1:  # If close to a minute boundary, wait for the next one
        initial_delay += 60
    libs.time.sleep(initial_delay)
    
    while not stop_event.is_set():
        elapsed_time = libs.time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        with open(filename, 'a') as f:
            f.write(f"{time_str:<11} {num_chunks:>10}  {initial_particles:>16}  {added_particles:>14}  {num_cpu:>7}  ({i_max}X{j_max})\n")
        libs.time.sleep(60)
    
    elapsed_time = libs.time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    with open(filename, 'a') as f:
        f.write(f"{time_str:<11} {num_chunks:>10}  {initial_particles:>16}  {added_particles:>14}  {num_cpu:>7}  ({i_max}X{j_max})\n")
        
def add_circle(ax, fill_black=False, zorder=0):
    """
    Add circles to the plot, including a GEO boundary circle, with a specified z-order.

    Parameters:
    - ax: Matplotlib axes object to plot on.
    - fill_black: Boolean to fill the right (negative X) side with black (default: False).
    - zorder: Z-order for the patches (default: 0).
    """
    # Draw the circle outline
    circle = libs.patches.Circle((0, 0), 1, edgecolor='white', facecolor='none', linewidth=2, zorder=zorder)
    GEO_circle = libs.patches.Circle((0, 0), 6.6, edgecolor='black', linestyle='-', facecolor='none', linewidth=2, zorder=zorder)
    ax.add_patch(circle)
    ax.add_patch(GEO_circle)
    if fill_black:
        # Fill the right (negative X) side with black
        circle_black = libs.patches.Wedge((0, 0), 1, 90, 270, edgecolor='none', facecolor='black', zorder=zorder)
        ax.add_patch(circle_black)
    
    # Add a dummy Line2D object for the legend to represent GEO_circle as a simple line
    geo_legend = libs.Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='GEO Boundary')
    ax.add_line(geo_legend)  # Add to the axes so it can be picked up by the legend
    
def plot_ellipse_boundary(a1=params.a1, a2=params.a2, semiminor=params.semiminor, y0=0.0, num_points=50, color='red', linestyle='-', plot=True, return_mask=False, ax=None):
    # Compute ellipse parameters
    semimajor = 0.5 * (a1 - a2)
    x0 = 0.5 * (a1 + a2)  # Ellipse center X

    # Validate ellipse parameters
    if semimajor <= 0:
        raise ValueError(f"Semi-major axis must be positive, got {semimajor} (from a1={a1}, a2={a2}).")
    if semiminor <= 0:
        raise ValueError(f"Semi-minor axis must be positive, got {semiminor}.")

    # Generate ellipse points
    theta = libs.np.linspace(0, 2 * libs.np.pi, num_points)
    x_ellipse = x0 + semimajor * libs.np.cos(theta)
    y_ellipse = y0 + semiminor * libs.np.sin(theta)

    # Plot the ellipse if enabled
    if plot:
        if ax is None:
            libs.plt.plot(x_ellipse, y_ellipse, color=color, linestyle=linestyle, label="RCM Boundary")
        else:
            ax.plot(x_ellipse, y_ellipse, color=color, linestyle=linestyle, label="RCM Boundary")

    # Define mask function (checks if (x, y) is inside the ellipse)
    def is_inside_ellipse(x, y):
        return ((x - x0) ** 2 / semimajor ** 2) + ((y - y0) ** 2 / semiminor ** 2) <= 1

    if return_mask:
        return x0, y0, semimajor, semiminor, is_inside_ellipse
    
    return x0, y0, semimajor, semiminor
    
def interpolate_to_boundary(x0, y0, x1, y1, is_inside_ellipse, buffer_distance=params.buffer_distance):
    # Check for NaN in input positions
    if libs.np.isnan(x0) or libs.np.isnan(y0) or libs.np.isnan(x1) or libs.np.isnan(y1):
        #print(f"interpolate_to_boundary: NaN detected in input positions (x0={x0}, y0={y0}, x1={x1}, y1={y1}). Returning starting position.")
        return x0, y0

    left, right = 0.0, 1.0
    for _ in range(20):
        mid = (left + right) / 2
        x_mid = x0 + mid * (x1 - x0)
        y_mid = y0 + mid * (y1 - y0)
        if is_inside_ellipse(x_mid, y_mid):
            left = mid
        else:
            right = mid
    
    t = left
    x = x0 + t * (x1 - x0)
    y = y0 + t * (y1 - y0)
    
    # Safeguard against division by zero in t_buffer calculation
    distance = libs.np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    if distance < 1e-10:  # Avoid division by zero
        t_buffer = right + 0.1  # Fallback to a small step
    else:
        t_buffer = right + (0.1 / distance)
    x_buffer = x0 + t_buffer * (x1 - x0)
    y_buffer = y0 + t_buffer * (y1 - y0)
    
    # Check for NaN in final position
    if libs.np.isnan(x_buffer) or libs.np.isnan(y_buffer):
        #print(f"interpolate_to_boundary: NaN detected in final position (x_buffer={x_buffer}, y_buffer={y_buffer}). Returning starting position.")
        return x0, y0
    
    return x_buffer, y_buffer