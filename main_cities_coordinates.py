import math, random
import cv2


class City:
    def __init__(self, x=None, y=None):
        self.x = None
        self.y = None
        if x is not None:
            self.x = x
        else:
            self.x = int(random.random() * 200)
        if y is not None:
            self.y = y
        else:
            self.y = int(random.random() * 200)

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def distanceTo(self, city):
        xDistance = abs(self.getX() - city.getX())
        yDistance = abs(self.getY() - city.getY())
        distance = math.sqrt((xDistance * xDistance) + (yDistance * yDistance))
        return distance / 20

    def __repr__(self):
        return str(self.getX()) + ", " + str(self.getY())


class TourManager:
    destinationCities = []

    def addCity(self, city):
        self.destinationCities.append(city)

    def getCity(self, index):
        return self.destinationCities[index]

    def numberOfCities(self):
        return len(self.destinationCities)


class Tour:
    def __init__(self, tourmanager, tour=None):
        self.tourmanager = tourmanager
        self.tour = []
        self.fitness = 0.0
        self.distance = 0
        if tour is not None:
            self.tour = tour
        else:
            for i in range(0, self.tourmanager.numberOfCities()):
                self.tour.append(None)

    def __len__(self):
        return len(self.tour)

    def __getitem__(self, index):
        return self.tour[index]

    def __setitem__(self, key, value):
        self.tour[key] = value

    def __repr__(self):
        geneString = "Start -> "
        for i in range(0, self.tourSize()):
            geneString += str(self.getCity(i)) + " -> "
        geneString += "End"
        return geneString

    def generateIndividual(self):
        for cityIndex in range(0, self.tourmanager.numberOfCities()):
            self.setCity(cityIndex, self.tourmanager.getCity(cityIndex))
        random.shuffle(self.tour)

    def getCity(self, tourPosition):
        return self.tour[tourPosition]

    def setCity(self, tourPosition, city):
        self.tour[tourPosition] = city
        self.fitness = 0.0
        self.distance = 0

    def getFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.getDistance())
        return self.fitness

    def getDistance(self):
        if self.distance == 0:
            tourDistance = 0
            for cityIndex in range(0, self.tourSize()):
                fromCity = self.getCity(cityIndex)
                destinationCity = None
                if cityIndex + 1 < self.tourSize():
                    destinationCity = self.getCity(cityIndex + 1)
                else:
                    destinationCity = self.getCity(0)
                tourDistance += fromCity.distanceTo(destinationCity)
            self.distance = tourDistance
        return self.distance

    def tourSize(self):
        return len(self.tour)

    def containsCity(self, city):
        return city in self.tour


class Population:
    def __init__(self, tourmanager, populationSize, initialise):
        self.tours = []
        for i in range(0, populationSize):
            self.tours.append(None)

        if initialise:
            for i in range(0, populationSize):
                newTour = Tour(tourmanager)
                newTour.generateIndividual()
                self.saveTour(i, newTour)

    def __setitem__(self, key, value):
        self.tours[key] = value

    def __getitem__(self, index):
        return self.tours[index]

    def saveTour(self, index, tour):
        self.tours[index] = tour

    def getTour(self, index):
        return self.tours[index]

    def getFittest(self):
        fittest = self.tours[0]
        for i in range(0, self.populationSize()):
            if fittest.getFitness() <= self.getTour(i).getFitness():
                fittest = self.getTour(i)
        return fittest

    def populationSize(self):
        return len(self.tours)


class GA:
    def __init__(self, tourmanager, mutationRate=0.05, tournamentSize=5, elitism=True):
        self.tourmanager = tourmanager
        self.mutationRate = mutationRate
        self.tournamentSize = tournamentSize
        self.elitism = elitism

    def evolvePopulation(self, pop):
        newPopulation = Population(self.tourmanager, pop.populationSize(), False)
        elitismOffset = 0
        if self.elitism:
            newPopulation.saveTour(0, pop.getFittest())
            elitismOffset = 1

        for i in range(elitismOffset, newPopulation.populationSize()):
            parent1 = self.tournamentSelection(pop)
            parent2 = self.tournamentSelection(pop)
            child = self.crossover(parent1, parent2)
            newPopulation.saveTour(i, child)

        for i in range(elitismOffset, newPopulation.populationSize()):
            self.mutate(newPopulation.getTour(i))

        return newPopulation

    def crossover(self, parent1, parent2):
        child = Tour(self.tourmanager)

        startPos = int(random.random() * parent1.tourSize())
        endPos = int(random.random() * parent1.tourSize())

        for i in range(0, child.tourSize()):
            if startPos < endPos and i > startPos and i < endPos:
                child.setCity(i, parent1.getCity(i))
            elif startPos > endPos:
                if not (i < startPos and i > endPos):
                    child.setCity(i, parent1.getCity(i))

        for i in range(0, parent2.tourSize()):
            if not child.containsCity(parent2.getCity(i)):
                for ii in range(0, child.tourSize()):
                    if child.getCity(ii) == None:
                        child.setCity(ii, parent2.getCity(i))
                        break

        return child

    def mutate(self, tour):
        for tourPos1 in range(0, tour.tourSize()):
            if random.random() < self.mutationRate:
                tourPos2 = int(tour.tourSize() * random.random())

                city1 = tour.getCity(tourPos1)
                city2 = tour.getCity(tourPos2)

                tour.setCity(tourPos2, city1)
                tour.setCity(tourPos1, city2)

    def tournamentSelection(self, pop):
        tournament = Population(self.tourmanager, self.tournamentSize, False)
        for i in range(0, self.tournamentSize):
            randomId = int(random.random() * pop.populationSize())
            tournament.saveTour(i, pop.getTour(randomId))
        fittest = tournament.getFittest()
        return fittest

def getsolutionfromfittests(cities_coordinates, tourmanager):
    n_cities = len(cities_coordinates)
    population_size = 50
    # n_generations = 100 
    n_generations = int(2 ** ((n_cities + 3.5) / 3.5)) + n_cities

    random.seed(200)

    # Load the map
    map_original = cv2.imread("map_Paris.png")

    for x,y in cities_coordinates:
        cv2.circle(
            map_original,
            center=(x, y),
            # radius=10,
            radius=20,
            color=(0, 0, 255),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    cv2.imshow("map", map_original)
    # 여행지점 이미지를 PNG 형식으로 저장
    if n_cities == 20:
        image_name = 'result_images/map_original_{}.png'.format(n_cities)
        cv2.imwrite(image_name, map_original)

    # cv2.waitKey(0)

    # Initialize population
    pop = Population(tourmanager, populationSize=population_size, initialise=True)
    print("Initial distance: " + str(pop.getFittest().getDistance()))

    # Evolve population
    ga = GA(tourmanager)

    for i in range(n_generations):
        pop = ga.evolvePopulation(pop)

        fittest = pop.getFittest()

        map_result = map_original.copy()

        for j in range(1, n_cities):
            cv2.line(
                map_result,
                pt1=(fittest[j - 1].x, fittest[j - 1].y),
                pt2=(fittest[j].x, fittest[j].y),
                color=(255, 0, 0),
                thickness=3,
                lineType=cv2.LINE_AA,
            )

        generation_text = "Generation: {0}, n_cities : {1}".format(i + 1,n_cities)
        cv2.putText(
            map_result,
            org=(10, 25),
            text=generation_text,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=0,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        distance_text = "Distance: %.2fkm" % fittest.getDistance()
        cv2.putText(
            map_result,
            org=(10, 50),
            text=distance_text,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=0,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        # print('generation : {}, distance : {}, Solution : {} !'
        #       .format(generation_text, distance_text, pop.getFittest()))
        cv2.imshow("map", map_result)
        if cv2.waitKey(100) == ord("q"):
            break
    # 결과 이미지를 PNG 형식으로 저장
    image_name = 'result_images/output_{}.png'.format(n_cities)
    cv2.imwrite(image_name, map_result)

    # Print final results
    # print("Finished")
    final_distance = str(pop.getFittest().getDistance())
    # print("Final distance: " + final_distance)
    solution = str(pop.getFittest())
    # print("Solution: "+solution)
    
    # cv2.waitKey(0)
    return (n_cities,final_distance,solution)

if __name__ == "__main__":
    n_cities = 20
    cities_coordinates = list()
    result_fittest = list()

    # Setup cities and tour
    tourmanager = TourManager()

    start_fittest_count = 4
    for i in range(n_cities):
        x = random.randint(200, 2200)
        y = random.randint(200, 1400)
        tourmanager.addCity(City(x=x, y=y))
        cities_coordinates.append((x, y))
        if len(cities_coordinates) >= start_fittest_count:
            result_fittest.append(getsolutionfromfittests(cities_coordinates, tourmanager))


    for n_cities,final_distance,solution in result_fittest:
        print(n_cities,final_distance)
    import seaborn as sns

    # 산점도 그리기
    # sns.scatterplot(data=result_fittest, x='final_distance', y='n_cities')

    import numpy as np

    # 거리를 시간으로 변환한 리스트 생성
    result_fittest_hours = np.array([float(t[1]) for t in result_fittest]) / 60
    print('거리를 시간으로 변환 리스트 : \n{}'.format(result_fittest_hours))

    # # 찾고자 하는 값
    target_hour = 5
    result_fittest_hours_withabs = np.abs(result_fittest_hours - target_hour)
    print(result_fittest_hours_withabs)
    # # 제시된 값에 가장 근접한 값의 인덱스 찾기
    idx = (result_fittest_hours_withabs).argmin()

    # 가장 근접한 값 출력
    print('가장 근접값 절대값 : \n{}'.format(result_fittest_hours_withabs[idx]))
    print('가장 근접값 경로 포함 : \n{}'.format(result_fittest[idx]))

    pass