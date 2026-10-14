"""Quick local API smoke test. Run from package root after requirements are installed."""
from fastapi.testclient import TestClient
from api import app


def main():
    client = TestClient(app)
    r = client.get('/health')
    assert r.status_code == 200, r.text
    assert r.json()['demand_model_loaded'] and r.json()['stockout_model_loaded']

    r = client.get('/catalog')
    assert r.status_code == 200, r.text
    assert 'PHC-DEMO-001' in r.json()['phcs']
    assert len(r.json()['medicines']) > 0

    r = client.post('/forecast', json={
        'phc_id': 'PHC-DEMO-001', 'medicine_id': 'MED001',
        'horizon_days': 7, 'crisis': False, 'current_stock': 100,
    })
    assert r.status_code == 200, r.text
    payload = r.json()
    assert len(payload['forecast']) == 7
    assert len(payload['inventory_projection']) == 7
    assert 'Recommended_Order_Quantity' in payload['summary']

    r = client.post('/stockout-risk', json={
        'phc_id': 'PHC-DEMO-001', 'medicine_id': 'MED001', 'crisis': False,
    })
    assert r.status_code == 200, r.text
    assert 0 <= r.json()['uncalibrated_stockout_score'] <= 1

    r = client.post('/forecast', json={
        'phc_id': 'UNKNOWN', 'medicine_id': 'MED001', 'horizon_days': 7,
    })
    assert r.status_code == 404, r.text

    print('API SMOKE TEST PASSED: /health, /catalog, /forecast, /stockout-risk, unknown PHC/medicine validation')


if __name__ == '__main__':
    main()
