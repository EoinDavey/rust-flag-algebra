use flag_algebra::*;
use flags::Graph;
use sdp::Problem;

fn has_c4(g: &Graph) -> bool {
    for a in 0..g.size() {
        for b in 0..g.size() {
            if !g.edge(a, b) {
                continue
            }
            for c in 0..g.size() {
                if !g.edge(b, c) || c == a {
                    continue
                }
                for d in 0..g.size() {
                    if !(g.edge(c, d) && g.edge(d, a)) || d == b {
                        continue
                    }
                    return true
                }
            }
        }
    }
    false
}

pub fn main() {
    let n = 4;
    let b = Basis::<Graph>::new(n);
    let flags_with_c4: QFlag<f64, _> = b.qflag_from_indicator(|g, _| has_c4(g));
    let edge = flag(&Graph::new(2, &[(0, 1)]));
    let pb = Problem {
        ineqs: vec![
            total_sum_is_one(b),
            flags_are_nonnegative(b),
            flags_with_c4.at_most(0.),
        ],
        cs: b.all_cs(),
        obj: edge.expand(b),
    };

    let mut f = FlagSolver::new(pb, "c4-free");
    f.init();
    f.print_report(); // Write some informations in report.html
    let opt = -f.optimal_value.unwrap();
    println!("Optimal value: {}", opt); // Expect this to be 0.
}
