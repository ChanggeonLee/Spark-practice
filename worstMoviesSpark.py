from pyspark import SparkConf, SparkContext
from itertools import islice
import csv

def loadMovies():
  movies = {}
  with open("/home/maria_dev/Spark-practice/data/ml-latest-small/movies.csv", "rb") as f:
    reader = csv.reader(f, delimiter=',')
    next(reader)
    for row in reader:
      movies[int(row[0])] = row[1]
  return movies

def parseInput(line):
  fields = line.split(',')
  return (int(fields[1]), (float(fields[2]), 1.0))

if __name__ == "__main__":
  movies = loadMovies()
  path = "hdfs:///user/maria_dev/ml-latest-small/ratings.csv"

  # create spark context
  conf = SparkConf().setAppName("WorstMovies")
  sc = SparkContext(conf = conf)

  # create RDD from text file
  lines = sc.textFile(path)

  # skip header
  lines = lines.mapPartitionsWithIndex(
    lambda idx, it: islice(it, 1, None) if idx == 0 else it
  )

  # line --> (movieId, (rating, 1.0))
  ratings = lines.map(parseInput)

  # reduc to (movieId, (sumOfRating, countRating))
  sumAndCounts = ratings.reduceByKey(lambda m1, m2:(m1[0]+m2[0], m1[1]+m2[1]))

  # sumAndCount --> (movieId, avrageRating)
  avrageRatings = sumAndCounts.mapValues(lambda v: v[0]/v[1])

  # sort
  sortedMovies = avrageRatings.sortBy(lambda x : x[1])

  # top 10
  results = sortedMovies.take(10)

  for result in results:
    print(movies[result[0]], result[1])