use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CropSettings {
    pub rotation: Option<f32>,
    pub aspect_ratio: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExposureSettings {
    pub exposure: Option<f32>,
    pub contrast: Option<f32>,
    pub highlights: Option<f32>,
    pub shadows: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ColorSettings {
    pub temperature: Option<f32>,
    pub tint: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GrainSettings {
    pub amount: f32,
    pub size: String, // "fine", "medium", "coarse"
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GeometrySettings {
    pub vertical: Option<f32>,
    pub horizontal: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QuickFixAdjustments {
    pub crop: Option<CropSettings>,
    pub exposure: Option<ExposureSettings>,
    pub color: Option<ColorSettings>,
    pub grain: Option<GrainSettings>,
    pub geometry: Option<GeometrySettings>,
}

impl QuickFixAdjustments {
    pub fn new() -> QuickFixAdjustments {
        QuickFixAdjustments::default()
    }
}

#[wasm_bindgen]
pub fn process_frame(
    data: &mut [u8],
    width: u32,
    height: u32,
    adjustments: JsValue,
) -> Result<Vec<u8>, JsValue> {
    let adjustments: QuickFixAdjustments = serde_wasm_bindgen::from_value(adjustments)?;
    operations::process_frame_internal(data, width, height, &adjustments)
        .map_err(|e| JsValue::from_str(&e))
}

pub mod operations;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialization() {
        let json = r#"{
            "crop": {"rotation": 90.0},
            "exposure": {"exposure": 1.0},
            "grain": {"amount": 0.5, "size": "fine", "seed": 123}
        }"#;
        let adj: QuickFixAdjustments = serde_json::from_str(json).unwrap();
        assert_eq!(adj.crop.unwrap().rotation, Some(90.0));
        assert_eq!(adj.exposure.unwrap().exposure, Some(1.0));
        assert_eq!(adj.grain.as_ref().unwrap().amount, 0.5);
        assert_eq!(adj.grain.as_ref().unwrap().seed, Some(123));
    }

    #[test]
    fn test_deterministic_grain() {
        let width = 100;
        let height = 100;
        let mut data1 = vec![128u8; (width * height * 4) as usize];
        let mut data2 = vec![128u8; (width * height * 4) as usize];

        let mut adj = QuickFixAdjustments::default();
        adj.grain = Some(GrainSettings {
            amount: 0.5,
            size: "medium".to_string(),
            seed: Some(555),
        });

        let res1 = operations::process_frame_internal(&mut data1, width, height, &adj).unwrap();
        let res2 = operations::process_frame_internal(&mut data2, width, height, &adj).unwrap();

        assert_eq!(res1, res2, "Grain output should be identical for same seed");
    }

    #[test]
    fn test_pipeline_no_panic() {
        let width = 10;
        let height = 10;
        let mut data = vec![0u8; (width * height * 4) as usize];
        let adj = QuickFixAdjustments::default();
        let res = operations::process_frame_internal(&mut data, width, height, &adj);
        assert!(res.is_ok());
    }
}
