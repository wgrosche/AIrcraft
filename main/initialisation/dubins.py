"""
Testing script for 3D Dubins path planning with visualization

This script tests the 3D Dubins implementation with various scenarios and provides
interactive 3D visualization of the computed paths.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import sys
import os

# Add the parent directory to the path to import the dubins modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircraft.dubins.dubins3d import DubinsManeuver3D_constructor, compute_sampling, getLowerBound, getUpperBound
from aircraft.dubins.dubins2d import DubinsManeuver2D

def test_basic_3d_maneuver():
    """Test basic 3D Dubins maneuver computation"""
    print("Testing basic 3D Dubins maneuver...")
    
    # Initial configuration: [x, y, z, heading, pitch]
    qi = [0.0, 0.0, 0.0, 0.0, 0.0]
    # Final configuration
    qf = [10.0, 10.0, 5.0, math.pi/2, math.pi/6]
    
    rhomin = 2.0
    pitchlims = [-math.pi/4, math.pi/3]
    
    try:
        maneuver = DubinsManeuver3D_constructor(qi, qf, rhomin, pitchlims)
        print(f"✓ Maneuver computed successfully")
        print(f"  Path length: {maneuver.length:.3f}")
        print(f"  Number of segments: {len(maneuver.path)}")
        if len(maneuver.path) >= 2:
            print(f"  Horizontal maneuver case: {maneuver.path[0].maneuver.case}")
            print(f"  Vertical maneuver case: {maneuver.path[1].maneuver.case}")
        return maneuver
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def test_multiple_scenarios():
    """Test multiple 3D Dubins scenarios"""
    print("\nTesting multiple scenarios...")
    
    scenarios = [
        {
            'name': 'Ascending turn',
            'qi': [0.0, 0.0, 0.0, 0.0, 0.0],
            'qf': [5.0, 5.0, 3.0, math.pi/2, math.pi/6],
            'rhomin': 1.5,
            'pitchlims': [-math.pi/6, math.pi/3]
        },
        {
            'name': 'Descending turn',
            'qi': [0.0, 0.0, 5.0, 0.0, 0.0],
            'qf': [8.0, -3.0, 1.0, -math.pi/3, -math.pi/8],
            'rhomin': 2.0,
            'pitchlims': [-math.pi/4, math.pi/4]
        },
        {
            'name': 'Level flight with heading change',
            'qi': [0.0, 0.0, 2.0, 0.0, 0.0],
            'qf': [6.0, 4.0, 2.0, math.pi, 0.0],
            'rhomin': 1.0,
            'pitchlims': [-math.pi/6, math.pi/6]
        },
        {
            'name': 'Steep climb',
            'qi': [0.0, 0.0, 0.0, 0.0, 0.0],
            'qf': [3.0, 3.0, 8.0, math.pi/4, math.pi/3],
            'rhomin': 1.5,
            'pitchlims': [-math.pi/6, math.pi/2]
        }
    ]
    
    maneuvers = []
    for scenario in scenarios:
        print(f"\n  Testing: {scenario['name']}")
        try:
            maneuver = DubinsManeuver3D_constructor(
                scenario['qi'], scenario['qf'], 
                scenario['rhomin'], scenario['pitchlims']
            )
            print(f"    ✓ Success - Length: {maneuver.length:.3f}")
            maneuvers.append((scenario['name'], maneuver))
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            maneuvers.append((scenario['name'], None))
    
    return maneuvers

def test_bounds():
    """Test lower and upper bound computations"""
    print("\nTesting bounds computation...")
    
    qi = [0.0, 0.0, 0.0, 0.0, 0.0]
    qf = [10.0, 8.0, 5.0, math.pi/3, math.pi/6]
    rhomin = 2.0
    pitchlims = [-math.pi/4, math.pi/3]
    
    try:
        lower_bound = getLowerBound(qi, qf, rhomin, pitchlims)
        upper_bound = getUpperBound(qi, qf, rhomin, pitchlims)
        actual_maneuver = DubinsManeuver3D_constructor(qi, qf, rhomin, pitchlims)
        
        print(f"  Lower bound length: {lower_bound.length:.3f}")
        print(f"  Actual length: {actual_maneuver.length:.3f}")
        print(f"  Upper bound length: {upper_bound.length:.3f}")
        
        if lower_bound.length <= actual_maneuver.length <= upper_bound.length:
            print("  ✓ Bounds are consistent")
        else:
            print("  ✗ Bounds are inconsistent")
            
        return lower_bound, actual_maneuver, upper_bound
    except Exception as e:
        print(f"  ✗ Error in bounds computation: {e}")
        return None, None, None

def visualize_3d_path(maneuver, title="3D Dubins Path", ax=None):
    """Visualize a 3D Dubins path"""
    if maneuver is None or len(maneuver.path) < 2:
        print("Cannot visualize: invalid maneuver")
        return None
    
    # Generate sampling points
    points = compute_sampling(maneuver, numberOfSamples=200)
    points = np.array(points)
    
    if ax is None:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
    
    # Plot the path
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 'b-', linewidth=2, label='Path')
    
    # Mark start and end points
    ax.scatter([maneuver.qi[0]], [maneuver.qi[1]], [maneuver.qi[2]], 
               color='green', s=100, label='Start', marker='o')
    ax.scatter([maneuver.qf[0]], [maneuver.qf[1]], [maneuver.qf[2]], 
               color='red', s=100, label='End', marker='s')
    
    # Add direction arrows at start and end
    arrow_length = 1.0
    
    # Start arrow
    start_dir = np.array([
        arrow_length * math.cos(maneuver.qi[3]) * math.cos(maneuver.qi[4]),
        arrow_length * math.sin(maneuver.qi[3]) * math.cos(maneuver.qi[4]),
        arrow_length * math.sin(maneuver.qi[4])
    ])
    ax.quiver(maneuver.qi[0], maneuver.qi[1], maneuver.qi[2],
              start_dir[0], start_dir[1], start_dir[2],
              color='green', arrow_length_ratio=0.3, linewidth=2)
    
    # End arrow
    end_dir = np.array([
        arrow_length * math.cos(maneuver.qf[3]) * math.cos(maneuver.qf[4]),
        arrow_length * math.sin(maneuver.qf[3]) * math.cos(maneuver.qf[4]),
        arrow_length * math.sin(maneuver.qf[4])
    ])
    ax.quiver(maneuver.qf[0], maneuver.qf[1], maneuver.qf[2],
              end_dir[0], end_dir[1], end_dir[2],
              color='red', arrow_length_ratio=0.3, linewidth=2)
    
    # Add intermediate direction arrows
    step = len(points) // 10
    for i in range(0, len(points), step):
        if i < len(points) - 1:
            # Calculate direction from consecutive points
            if i + 1 < len(points):
                direction = points[i+1] - points[i]
                direction = direction / np.linalg.norm(direction) * 0.5
                ax.quiver(points[i, 0], points[i, 1], points[i, 2],
                         direction[0], direction[1], direction[2],
                         color='blue', alpha=0.6, arrow_length_ratio=0.3)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    # Set equal aspect ratio
    max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                         points[:, 1].max() - points[:, 1].min(),
                         points[:, 2].max() - points[:, 2].min()]).max() / 2.0
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    return ax

def visualize_multiple_scenarios(maneuvers):
    """Visualize multiple scenarios in subplots"""
    valid_maneuvers = [(name, m) for name, m in maneuvers if m is not None]
    
    if not valid_maneuvers:
        print("No valid maneuvers to visualize")
        return
    
    n_plots = len(valid_maneuvers)
    cols = 2
    rows = (n_plots + 1) // 2
    
    fig = plt.figure(figsize=(15, 6 * rows))
    
    for i, (name, maneuver) in enumerate(valid_maneuvers):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        visualize_3d_path(maneuver, title=name, ax=ax)
    
    plt.tight_layout()
    return fig

def analyze_path_properties(maneuver):
    """Analyze properties of the computed path"""
    if maneuver is None or len(maneuver.path) < 2:
        print("Cannot analyze: invalid maneuver")
        return
    
    print(f"\nPath Analysis:")
    print(f"  Total length: {maneuver.length:.3f}")
    print(f"  Start config: [{maneuver.qi[0]:.2f}, {maneuver.qi[1]:.2f}, {maneuver.qi[2]:.2f}, "
          f"{math.degrees(maneuver.qi[3]):.1f}°, {math.degrees(maneuver.qi[4]):.1f}°]")
    print(f"  End config: [{maneuver.qf[0]:.2f}, {maneuver.qf[1]:.2f}, {maneuver.qf[2]:.2f}, "
          f"{math.degrees(maneuver.qf[3]):.1f}°, {math.degrees(maneuver.qf[4]):.1f}°]")
    
    # Analyze horizontal component
    Dlat = maneuver.path[0]
    print(f"  Horizontal component:")
    print(f"    Case: {Dlat.maneuver.case}")
    print(f"    Length: {Dlat.maneuver.length:.3f}")
    print(f"    Segments: t={Dlat.maneuver.t:.3f}, p={Dlat.maneuver.p:.3f}, q={Dlat.maneuver.q:.3f}")
    
    # Analyze vertical component
    Dlon = maneuver.path[1]
    print(f"  Vertical component:")
    print(f"    Case: {Dlon.maneuver.case}")
    print(f"    Length: {Dlon.maneuver.length:.3f}")
    print(f"    Segments: t={Dlon.maneuver.t:.3f}, p={Dlon.maneuver.p:.3f}, q={Dlon.maneuver.q:.3f}")
    
    # Calculate actual distance
    euclidean_dist = np.linalg.norm(maneuver.qf[:3] - maneuver.qi[:3])
    print(f"  Euclidean distance: {euclidean_dist:.3f}")
    print(f"  Path efficiency: {euclidean_dist/maneuver.length:.3f}")

def interactive_demo():
    """Interactive demonstration allowing user to modify parameters"""
    print("\n" + "="*50)
    print("INTERACTIVE 3D DUBINS DEMO")
    print("="*50)
    
    while True:
        print("\nEnter configuration (or 'q' to quit):")
        
        try:
            # Get user input
            qi_input = input("Start [x y z heading(deg) pitch(deg)]: ").strip()
            if qi_input.lower() == 'q':
                break
                
            qf_input = input("End [x y z heading(deg) pitch(deg)]: ").strip()
            if qf_input.lower() == 'q':
                break
            
            rhomin_input = input("Min turning radius (default 2.0): ").strip()
            rhomin = float(rhomin_input) if rhomin_input else 2.0
            
            pitch_input = input("Pitch limits [min_deg max_deg] (default -45 60): ").strip()
            if pitch_input:
                pitch_vals = list(map(float, pitch_input.split()))
                pitchlims = [math.radians(pitch_vals[0]), math.radians(pitch_vals[1])]
            else:
                pitchlims = [math.radians(-45), math.radians(60)]
            # Convert inputs to appropriate types            # Parse configurations
            qi_vals = list(map(float, qi_input.split()))
            qf_vals = list(map(float, qf_input.split()))
            
            # Convert angles to radians
            qi = [qi_vals[0], qi_vals[1], qi_vals[2], 
                  math.radians(qi_vals[3]), math.radians(qi_vals[4])]
            qf = [qf_vals[0], qf_vals[1], qf_vals[2], 
                  math.radians(qf_vals[3]), math.radians(qf_vals[4])]
            
            # Compute maneuver
            print("Computing 3D Dubins path...")
            maneuver = DubinsManeuver3D_constructor(qi, qf, rhomin, pitchlims)
            
            # Analyze and visualize
            analyze_path_properties(maneuver)
            
            # Create visualization
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            visualize_3d_path(maneuver, "Interactive 3D Dubins Path", ax)
            plt.show()
            
        except ValueError as e:
            print(f"Invalid input format: {e}")
        except Exception as e:
            print(f"Error computing path: {e}")

def benchmark_performance():
    """Benchmark the performance of 3D Dubins computation"""
    print("\n" + "="*50)
    print("PERFORMANCE BENCHMARK")
    print("="*50)
    
    import time
    
    # Generate random test cases
    np.random.seed(42)  # For reproducible results
    n_tests = 100
    
    test_cases = []
    for _ in range(n_tests):
        qi = [
            np.random.uniform(-10, 10),  # x
            np.random.uniform(-10, 10),  # y
            np.random.uniform(0, 10),    # z
            np.random.uniform(0, 2*math.pi),  # heading
            np.random.uniform(-math.pi/6, math.pi/6)  # pitch
        ]
        qf = [
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10),
            np.random.uniform(0, 10),
            np.random.uniform(0, 2*math.pi),
            np.random.uniform(-math.pi/6, math.pi/6)
        ]
        rhomin = np.random.uniform(1.0, 3.0)
        pitchlims = [-math.pi/4, math.pi/3]
        
        test_cases.append((qi, qf, rhomin, pitchlims))
    
    # Run benchmark
    successful_computations = 0
    total_time = 0
    path_lengths = []
    
    print(f"Running {n_tests} random test cases...")
    
    start_time = time.time()
    for i, (qi, qf, rhomin, pitchlims) in enumerate(test_cases):
        try:
            case_start = time.time()
            maneuver = DubinsManeuver3D_constructor(qi, qf, rhomin, pitchlims)
            case_time = time.time() - case_start
            
            successful_computations += 1
            total_time += case_time
            path_lengths.append(maneuver.length)
            
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{n_tests} cases...")
                
        except Exception as e:
            print(f"  Case {i+1} failed: {e}")
    
    total_benchmark_time = time.time() - start_time
    
    # Report results
    print(f"\nBenchmark Results:")
    print(f"  Total cases: {n_tests}")
    print(f"  Successful: {successful_computations}")
    print(f"  Success rate: {successful_computations/n_tests*100:.1f}%")
    print(f"  Total time: {total_benchmark_time:.3f}s")
    print(f"  Average time per case: {total_time/successful_computations*1000:.2f}ms")
    print(f"  Average path length: {np.mean(path_lengths):.3f}")
    print(f"  Path length std: {np.std(path_lengths):.3f}")

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print("\n" + "="*50)
    print("EDGE CASE TESTING")
    print("="*50)
    
    edge_cases = [
        {
            'name': 'Same start and end position',
            'qi': [0.0, 0.0, 0.0, 0.0, 0.0],
            'qf': [0.0, 0.0, 0.0, math.pi, 0.0],
            'rhomin': 1.0,
            'pitchlims': [-math.pi/4, math.pi/4]
        },
        {
            'name': 'Very small turning radius',
            'qi': [0.0, 0.0, 0.0, 0.0, 0.0],
            'qf': [10.0, 10.0, 5.0, math.pi/2, math.pi/6],
            'rhomin': 0.1,
            'pitchlims': [-math.pi/4, math.pi/4]
        },
        {
            'name': 'Very large turning radius',
            'qi': [0.0, 0.0, 0.0, 0.0, 0.0],
            'qf': [5.0, 5.0, 2.0, math.pi/4, math.pi/8],
            'rhomin': 10.0,
            'pitchlims': [-math.pi/4, math.pi/4]
        },
        {
            'name': 'Tight pitch constraints',
            'qi': [0.0, 0.0, 0.0, 0.0, 0.0],
            'qf': [5.0, 5.0, 8.0, math.pi/2, math.pi/3],
            'rhomin': 2.0,
            'pitchlims': [-math.pi/12, math.pi/12]  # Very tight: ±15 degrees
        },
        {
            'name': 'Opposite directions',
            'qi': [0.0, 0.0, 0.0, 0.0, 0.0],
            'qf': [1.0, 0.0, 0.0, math.pi, 0.0],
            'rhomin': 2.0,
            'pitchlims': [-math.pi/4, math.pi/4]
        }
    ]
    
    for case in edge_cases:
        print(f"\nTesting: {case['name']}")
        try:
            maneuver = DubinsManeuver3D_constructor(
                case['qi'], case['qf'], case['rhomin'], case['pitchlims']
            )
            print(f"  ✓ Success - Length: {maneuver.length:.3f}")
            
            # Quick validation
            if maneuver.length < 0:
                print(f"  ⚠ Warning: Negative path length")
            if math.isinf(maneuver.length):
                print(f"  ⚠ Warning: Infinite path length")
                
        except Exception as e:
            print(f"  ✗ Failed: {e}")

def main():
    """Main testing function"""
    print("3D DUBINS PATH PLANNING - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    # Run all tests
    basic_maneuver = test_basic_3d_maneuver()
    multiple_maneuvers = test_multiple_scenarios()
    lower_bound, actual, upper_bound = test_bounds()
    test_edge_cases()
    
    # Visualizations
    print("\n" + "="*50)
    print("VISUALIZATION")
    print("="*50)
    
    if basic_maneuver:
        print("Displaying basic maneuver...")
        fig1 = plt.figure(figsize=(12, 9))
        ax1 = fig1.add_subplot(111, projection='3d')
        visualize_3d_path(basic_maneuver, "Basic 3D Dubins Maneuver", ax1)
        
        analyze_path_properties(basic_maneuver)
    
    if multiple_maneuvers:
        print("Displaying multiple scenarios...")
        fig2 = visualize_multiple_scenarios(multiple_maneuvers)
    
    if all([lower_bound, actual, upper_bound]):
        print("Displaying bounds comparison...")
        fig3 = plt.figure(figsize=(15, 5))
        
        ax1 = fig3.add_subplot(131, projection='3d')
        visualize_3d_path(lower_bound, "Lower Bound", ax1)
        
        ax2 = fig3.add_subplot(132, projection='3d')
        visualize_3d_path(actual, "Actual Path", ax2)
        
        ax3 = fig3.add_subplot(133, projection='3d')
        visualize_3d_path(upper_bound, "Upper Bound", ax3)
        
        plt.tight_layout()
    
    # Performance benchmark
    benchmark_performance()
    
    # Show all plots
    plt.show()
    
    # Interactive demo
    response = input("\nWould you like to try the interactive demo? (y/n): ")
    if response.lower().startswith('y'):
        interactive_demo()

if __name__ == "__main__":
    main()
