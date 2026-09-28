from quantylab.trainer.etf_single_intraday.microstructure_audit import extract


def test_uses_joint_receipt_and_latest_intensity():
    r=dict(code='102110',quote_received_at='2026-09-14T09:01:00+09:00',
        intensity_received_at='2026-09-14T09:01:02+09:00',quote={},
        intensity=[{'체결시간':'090000','체결강도':10},{'체결시간':'090100','체결강도':20},
                   {'체결시간':'090200','체결강도':999}])
    result=extract(r)
    assert result['at'].second==2
    assert result['intensity']==20
    assert result['intensity_age']==2
