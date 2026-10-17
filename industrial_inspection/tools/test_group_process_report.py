"""验证结束汇总中的未确认、未出现、已关闭和未退出群组，不依赖模型推理。"""
import json
import unittest

from industrial_inspection.tools import predict_slot_groups as predict


def snapshot(group_id,sequence=(),states=('empty',)*4,phase='unload',order='pending'):
    target='empty' if phase=='unload' else 'filled'
    return {'group_id':group_id,'phase':phase,'confirmed_event_sequence':list(sequence),
            'slots':[{'slot_id':i+1,'last_confirmed_state':state} for i,state in enumerate(states)],
            'all_four_target_previously_verified':all(state==target for state in states),
            'order_result':order}


def summary(current,closed=(),processed=10,status='end_of_stream'):
    return {'phase':current['phase'],'group_count':5,'processed_frames':processed,'status':status,
            'closed_groups':[{'group_id':s['group_id'],'snapshot':s} for s in closed],
            'final_group_snapshot':current}


class GroupProcessReportTests(unittest.TestCase):
    def test_partial_current_group_and_unseen_groups_have_distinct_matrix_cells(self):
        first=snapshot(1,(2,1,3,4),order='pass')
        second=snapshot(2,(2,1),states=('empty','empty','filled','unknown'))
        report=predict.build_process_report(summary(second,[first]))
        self.assertEqual(report['matrix'],[[1,1,1,1],[1,1,0,0],[None]*4,[None]*4,[None]*4])
        self.assertTrue(report['groups'][0]['completed'])
        self.assertFalse(report['groups'][1]['completed'])
        self.assertFalse(report['groups'][2]['observed'])
        self.assertEqual(report['groups'][1]['sequence'],[2,1])
        json.dumps(report,allow_nan=False)

    def test_initially_empty_without_unload_evidence_is_not_completed(self):
        current=snapshot(1)
        report=predict.build_process_report(summary(current))
        self.assertEqual(report['matrix'][0],[0,0,0,0])
        self.assertFalse(predict.process_completed(current))

    def test_refilled_slot_is_not_marked_completed(self):
        current=snapshot(1,(2,1,3,4),states=('empty','empty','empty','filled'))
        report=predict.build_process_report(summary(current))
        self.assertEqual(report['matrix'][0],[1,1,1,0])
        self.assertFalse(report['groups'][0]['completed'])
        self.assertFalse(predict.process_completed(current))

    def test_closed_snapshot_is_not_replaced_by_final_waiting_state(self):
        closed=snapshot(5,(2,1,3,4),order='pass')
        current=snapshot(5,states=('unknown',)*4)
        report=predict.build_process_report(summary(current,[closed]))
        self.assertEqual(report['matrix'][4],[1,1,1,1])
        self.assertEqual(report['groups'][4]['sequence'],[2,1,3,4])

    def test_zero_processed_frames_does_not_claim_group_was_seen(self):
        report=predict.build_process_report(summary(snapshot(1),processed=0,status='user_stopped'))
        self.assertEqual(report['matrix'],[[None]*4]*5)

    def test_completion_and_order_validation_are_separate(self):
        current=snapshot(1,(1,2,3,4),order='fail')
        report=predict.build_process_report(summary(current))
        self.assertTrue(report['groups'][0]['completed'])
        self.assertEqual(report['groups'][0]['order_result'],'fail')
        self.assertTrue(predict.process_completed(current))

    def test_load_phase_and_early_stop_are_labeled_correctly(self):
        current=snapshot(3,(1,3,4),states=('filled','empty','filled','filled'),phase='load')
        report=predict.build_process_report(summary(current,status='frame_limit'))
        self.assertEqual(report['matrix'][2],[1,0,1,1])
        text=predict.format_process_report(report)
        self.assertIn('游星片槽位上料完成矩阵',text)
        self.assertIn('G3：1 → 3 → 4',text)
        self.assertIn('G1：未记录',text)
        self.assertIn('已处理视频范围',text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
