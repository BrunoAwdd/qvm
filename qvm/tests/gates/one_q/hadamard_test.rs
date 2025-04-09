use qlang::gates::one_q::hadamard::Hadamard;
use qlang::qvm::QVM;
use qlang::qlang::QLang;
use qlang::types::qlang_complex::QLangComplex;
use ndarray::array;

#[test]
fn test_hadamard_matrix() {
    let h = Hadamard::new();
    let factor = 1.0 / (2.0_f64).sqrt();

    let expected = array![
        [QLangComplex::new(factor, 0.0), QLangComplex::new(factor, 0.0)],
        [QLangComplex::new(factor, 0.0), QLangComplex::new(-factor, 0.0)]
    ];

    assert_eq!(h.matrix, expected);
}

#[test]
fn test_hadamard_apply() {
    let mut qvm = QVM::new(1);
    qvm.apply_gate(&Hadamard::new(), 0);

    let mut count = 0;
    for _ in 0..10 {
        let mut copy = qvm.clone();
        count += copy.measure(0) as usize;
    }

    println!("Probabilidade de 1: {}", count as f64 / 10.0);
}

#[test]
fn test_hadamard_distribution() {
    let mut count_0 = 0;
    let mut count_1 = 0;
    let iterations = 100;

    for _ in 0..iterations {
        let mut qlang = QLang::new(1);
        qlang.run_qlang_from_line("h(0)").unwrap();

        qlang.run_qlang_from_line("run()").unwrap();

        let result = qlang.qvm.measure(0);

        if result == 0 {
            count_0 += 1;
        } else {
            count_1 += 1;
        }
    }

    println!("0s: {}, 1s: {}", count_0, count_1);

    let prob_0 = count_0 as f64 / iterations as f64;
    let prob_1 = count_1 as f64 / iterations as f64;

    println!("Prob 0: {:.3}, Prob 1: {:.3}", prob_0, prob_1);

    // Assert that both probabilities are roughly close to 0.5 (±10%)
    assert!((0.4..=0.6).contains(&prob_0), "Probabilidade de 0 fora do esperado");
    assert!((0.4..=0.6).contains(&prob_1), "Probabilidade de 1 fora do esperado");
}
#[test]
fn test_debug_hadamard() {
    let mut qlang = QLang::new(1);

    //qlang.run_qlang_from_line("create(1)").unwrap();
    //println!("AST após create: {:?}", qlang.ast);

    qlang.run_qlang_from_line("h(0)").unwrap();
    println!("AST após h(0): {:?}", qlang.ast);

    qlang.run_qlang_from_line("run()").unwrap();

    qlang.qvm.display(); // opcional: ver estado antes da medição

    let result = qlang.qvm.measure(0);
    println!("Resultado final da medida: {}", result);
}
#[test]
fn test_measure_many_hadamard() {
    let mut qlang = QLang::new(3);
    let m0 = qlang.run_qlang_from_line("measure(0, 1, 2)").unwrap(); // Medições múltiplas
    //qlang.run_qlang_from_line("create(1)").unwrap();
    qlang.run_qlang_from_line("h(0)").unwrap();
    let m1 = qlang.run_qlang_from_line("measure(0, 1, 2)").unwrap(); // Medições múltiplas
    qlang.run_qlang_from_line("x(1)").unwrap(); // Aplica o PX
    let m2 = qlang.run_qlang_from_line("measure(0, 1, 2)").unwrap(); // Medições múltiplas

    let m_all = qlang.run_qlang_from_line("measure_all()").unwrap();

    qlang.run(); // Executa AST até esse ponto

    let result = qlang.qvm.measure_all();

    println!("M0 = {:?}, M1 = {:?}, M2 = {:?}", m0, m1, m2);
    println!("M_ALL = {:?}", m_all);
    println!("Resultado final do qubit 0: {:?}", result);

    assert!(result[0] == 0 || result[0] == 1, "Resultado inválido de medição");

}
