from utils.diabetes_xgb_praktikum import build_comparison_bundle
b = build_comparison_bundle(save_model=False)
print(b['comparison_table'].to_string(index=False))
print('best:', b['best_result_key'])
