use std::collections::HashMap;
use std::path::Path;
use std::ffi::OsStr;
use libloading::{Library, Symbol};
use std::sync::Arc;
use super::quantum_gate_abstract::QuantumGateAbstract;

pub struct PortalRegistry {
    pub gates: HashMap<String, Arc<dyn QuantumGateAbstract>>,
    _libs: Vec<Library>, // manter as libs vivas
}

impl PortalRegistry {
pub fn new() -> Self {
        Self {
            gates: HashMap::new(),
            _libs: vec![],
        }
    }

    pub fn register(&mut self, gate: Arc<dyn QuantumGateAbstract>) {
        let name = gate.name().to_lowercase();
        self.gates.insert(name, gate);
    }

    pub fn list(&self) -> Vec<String> {
        self.gates.keys().cloned().collect()
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn QuantumGateAbstract>> {
        self.gates.get(name).cloned()
    }

    pub fn load_from_dir(path: &Path) -> Result<Self, String> {
        let mut registry = Self::new();

        for entry in std::fs::read_dir(path).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let lib_path = entry.path();

            if lib_path.extension() == Some(OsStr::new("so")) {
                unsafe {
                    let lib = Library::new(&lib_path).map_err(|e| e.to_string())?;

                    // Signature: extern "C" fn(&mut PortalRegistry)
                    let func: Symbol<unsafe extern "C" fn(&mut PortalRegistry)> =
                        lib.get(b"register_with").map_err(|e| e.to_string())?;

                    func(&mut registry);
                    registry._libs.push(lib); // mantemos a lib viva
                }
            }
        }

        Ok(registry)
    }
}
