from pyspark.sql import SparkSession, Row
from typing import Tuple
import re
from argparse import ArgumentParser


str_value = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7
}

black_player_pattern = r'PB\[[a-zA-Z0-9]*\]'
white_player_pattern = r'PW\[[a-zA-Z0-9]*\]'
black_rate_pattern = r'RB\[-?\d+(\.\d+)?\]'
white_rate_pattern = r'RW\[-?\d+(\.\d+)?\]'
result_pattern = r'RE\[-?\d+(\.\d+)?\]'
move_pattern = r'([BW]\[(?:(?:[A-H][1-8])|pass)(?://|/)[\d.-]+(?:/[\d.-]+)?\])'

# e.g. W[D2//0.92]


def to_hand(s: str):
    turn = 1 if s[0] == "B" else -1
    move = s[2:].split('/')[0]
    if move == 'pass':
        return turn, -1

    horizonal = str_value[move[0]] * 8
    vertical = int(move[1]) - 1
    return turn, horizonal + vertical


def hand_to_str(hand: Tuple[int, int]):
    return str(hand[0]) + "#" + str(hand[1])


def parse_line(line: str):
    # extract player
    black_player = re.search(black_player_pattern, line).group()[3:-1]
    white_player = re.search(white_player_pattern, line).group()[3:-1]
    # extract rating
    black_rate = float(re.search(black_rate_pattern, line).group()[3:-1])
    white_rate = float(re.search(white_rate_pattern, line).group()[3:-1])

    # extract result
    result = re.search(result_pattern, line).group()
    result = float(result[3:-1])
    result = 1 if result > 0 else (0 if result == 0 else -1)

    # extract moves
    moves = re.findall(move_pattern, line)
    length = len(moves)

    moves = map(lambda s: hand_to_str(to_hand(s)), moves)
    moves = ",".join(moves)

    return Row(
        black_player=black_player,
        white_player=white_player,
        black_rate=black_rate,
        white_rate=white_rate,
        result=result,
        moves=moves,
        length=length
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('file', help='input ggf file path')
    parser.add_argument('postgres_host', metavar='h',
                        default='localhost', help='postgres host for data store')
    parser.add_argument('postgres_port', metavar='p',
                        default=5432, help='postgres port')
    args = parser.parse_args()

    # read ggf file
    spark = SparkSession.Builder().appName("PostgresPipeline").getOrCreate()
    ggf = spark.sparkContext.textFile(args.file)

    df = ggf.map(parse_line).toDF()

    df.write \
      .format("jdbc") \
      .option("url", f"jdbc:postgresql://{args.postgres_host}:{args.postgres_port}/reversi") \
      .option("dbtable", "history") \
      .option("driver", "org.postgresql.Driver") \
      .save()

    spark.stop()
