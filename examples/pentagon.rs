use flag_algebra::*;
use flags::Graph;
use operator::Basis;
use sdp::Problem;

pub fn main() {
    init_default_log();

    // Work on the graphs of size 3.
    let basis = Basis::new(5);

    // Define useful flags.
    let triangle = flag(&Graph::new(3, &[(0, 1), (1, 2), (2, 0)]));
    let c5 = flag(&Graph::new(5, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]));

    // Definition of the optimization problem.
    let pb = Problem::<f64, _> {
        // Constraints
        ineqs: vec![
            total_sum_is_one(basis),
            flags_are_nonnegative(basis),
            triangle.expand(basis).at_most(0.0),
        ],
        // Use all relevant Cauchy-Schwarz inequalities.
        cs: basis.all_cs(),
        obj: -c5.expand(basis),
    }.no_scale();

    let mut f = FlagSolver::new(pb, "pentagon");
    f.init();
    f.print_report(); // Write some informations in report.html

    let result = -f.optimal_value.expect("Failed to get optimal value");

    println!("Optimal value: {}", result); // Must equal 24/625 = 0.0384
}
