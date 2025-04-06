use std::fmt;

#[derive(Clone, Debug)]
pub enum QLangCommand {
    Create(usize),
    ApplyGate(String, Vec<String>),
    MeasureAll,
    Measure(usize),
    MeasureMany(Vec<usize>),
    Display,
}
impl fmt::Display for QLangCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QLangCommand::Create(n) => writeln!(f, "create({})", n),
            QLangCommand::ApplyGate(name, args) => {
                println!("Gate Name:{}({})", name, args.join(","));
                writeln!(f, "{}({})", name, args.join(","))
            }
            QLangCommand::MeasureAll => writeln!(f, "measure_all()"),
            QLangCommand::Measure(q) => writeln!(f, "measure({})", q),
            QLangCommand::MeasureMany(qs) => {
                let list = qs.iter().map(|q| q.to_string()).collect::<Vec<_>>().join(",");
                writeln!(f, "measure({})", list)
            }
            QLangCommand::Display => writeln!(f, "display()"),
        }
    }
}
