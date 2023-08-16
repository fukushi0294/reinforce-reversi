import unittest
from spark_pipeline import parse_line, to_hand

class TestGGFParseMethod(unittest.TestCase):
    def test_parse_line(self):
        text = "(;GM[Othello]PC[GGS/os]DT[940606712]PB[igor]PW[ant]RB[1285.87]RW[1471.98]TI[15:00//02:00]TY[8]RE[-38.00]BO[8 -------- -------- -------- ---O*--- ---*O--- -------- -------- -------- *]B[D3//4.58]W[C5/8.57/1.56]B[D6//1.67]W[E3/-35.58/0.09]B[F4//1.05]W[C6/-52.58/0.08]B[F5//1.41]W[C3/-24.58/0.06]B[C4//1.00]W[B5/-39.58/0.06]B[B4//0.93]W[F3/18.69/0.02]B[A5//2.25]W[B6/8.83/0.01]B[E6//1.40]W[A6/3.56]B[A7//1.72]W[A3/5.22/0.01]B[C7//2.88]W[D7/8.34/0.01]B[C8//1.78]W[G4/5.74/0.01]B[H3//4.70]W[F7/8.15]B[E7//3.44]W[G6/4.89/0.01]B[G3//3.31]W[G5/8.55/0.01]B[H5//1.46]W[F8/18.78/0.01]B[E8//4.44]W[F6/20.19/0.01]B[F2//4.39]W[D2/22.85/0.02]B[E2//1.56]W[C2/24.21/0.01]B[D1//4.15]W[B3/27.01/0.01]B[C1//2.23]W[A4/29.50/0.03]B[A2//1.40]W[G2/31.79/0.01]B[F1//4.67]W[H1/36.59/0.01]B[G1//1.46]W[E1/42.76]B[B2//1.40]W[A1/35.63/0.01]B[B1//0.88]W[H2/34.88/0.01]B[G8//3.04]W[D8/35.22/0.01]B[G7//1.66]W[H8/35.13/0.01]B[H7//0.91]W[H6/37.81/0.01]B[pass//0.68]W[H4/40.62/0.01]B[pass//0.62]W[B7/39.05/0.01]B[B8//1.80]W[A8];)"
        row = parse_line(text)
        self.assertEqual(row.black_player, "igor")
        self.assertEqual(row.white_player, "ant")
        self.assertEqual(row.black_rate, 1285.87)
        self.assertEqual(row.white_rate, 1471.98)
        self.assertEqual(row.result, -1)
        self.assertEqual(row.length, 61)
    
    def test_to_hand(self):
        text = "W[D2//0.92]"
        turn, hand = to_hand(text)
        self.assertEqual(turn, -1)
        self.assertEqual(hand, 25)
    
    def test_to_hand_when_pass(self):
        text = "B[pass//]"
        turn, hand = to_hand(text)
        self.assertEqual(turn, 1)
        self.assertEqual(hand, -1)

