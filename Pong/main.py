import os
import pickle
import pygame
import neat
import time

from neat import Population

from pong import Game


class PongGame:
    def __init__(self, window, width, height):
        # Inicjalizacja gry
        self.game = Game(window, width, height)
        self.genome1 = None
        self.genome2 = None
        self.ball = self.game.ball
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle

    def testing_ai(self, net):
        # Testowanie AI
        clock = pygame.time.Clock()
        run = True

        while run:
            # FPS limiter
            clock.tick(60)
            # Przechwytywanie eventów
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            output = net.activate((self.right_paddle.y, abs(self.right_paddle.x - self.ball.x), self.ball.y))
            decision = max(enumerate(output), key=lambda x: x[1])[0]

            if decision == 1:
                self.game.move_paddle(left=False, up=True)
            elif decision == 2:
                self.game.move_paddle(left=False, up=False)

            self.game.draw(draw_score=True)
            pygame.display.update()

    def train_single_ai_pair(self, g1, g2, cfg, draw):
        # Trenowanie SI przy użyciu dwóch genomów NEAT i konfiguracji
        start_time = time.time()
        # FeedForward nie jest rekurencyjną siecią co oznacza że nie ma pamięci wstecznej
        n1 = neat.nn.FeedForwardNetwork.create(g1, cfg)
        n2 = neat.nn.FeedForwardNetwork.create(g2, cfg)
        self.genome1 = g1
        self.genome2 = g2
        max_hits = 25

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True

            game_info = self.game.loop()
            # call move by AI
            self.move_ai_paddles(n1, n2)

            if draw:
                self.game.draw(draw_score=True, draw_hits=True)

            pygame.display.update()

            duration = time.time() - start_time
            if game_info.left_score == 3 or game_info.right_score == 3 or game_info.left_hits >= max_hits:  # max hits == 50
                self.calculate_fitness(game_info, duration)
                break

        return False

    def move_ai_paddles(self, n1, n2):
        # Poruszanie przez AI paletkami na podstawie podanych sieci neuronowych
        players = [
            (self.genome1, n1, self.left_paddle, True),
            (self.genome2, n2, self.right_paddle, False)
        ]

        for genome, net, paddle, left in players:
            output = net.activate((paddle.y, abs(paddle.x - self.ball.x), self.ball.y))
            decision = max(enumerate(output), key=lambda x: x[1])[0]

            if decision == 0:  # Nie poruszaj
                genome.fitness -= 0.01  # Zmniejszanie wartości przystosowania dla tej decyzji
            else:  # Porusz w górę lub w dół
                valid = self.game.move_paddle(left=left, up=(decision == 1))

                if not valid:  # jeśli paletka wychodzi poza ekran
                    genome.fitness -= 1

    def calculate_fitness(self, game_info, duration):
        # Obliczanie dokładności genomów na podstawie informacji o grze
        # Fitness to prościej ujmując wynik. W przypadku poniżej
        # self.genome1.fitness += game_info.left_score + game_info.left_hits - duration
        # self.genome2.fitness += game_info.right_score + game_info.right_hits - duration
        self.genome1.fitness += game_info.left_hits + duration
        self.genome2.fitness += game_info.right_hits + duration
        # pierwsza próba - ostatnie generacje dochodziły do wniosku że czas + ilość uderzeń była
        # najbardziej korzystna dla obu stron więc odbijały w jedym miejscu oby dwie przez max'a wiecznie


def eval_genomes(genomes, cfg):
    # Ewaluacja genomów
    # Gen - Node lub connection, reprezentuje różne cechy, takie jak wagi połączeń między neuronami, progi aktywacji
    # Genom - set Genów
    width, height = 700, 500
    win = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pong")
    # iterate przez index oraz wartość genomu wszystkich genomów
    for i, (genome_id1, genome1) in enumerate(genomes):
        # zerowanie przystosowania aby uniknąć nierównych szans podczas
        genome1.fitness = 0
        # wybieranie podzbioru w populacji, aby każdy miał szanse zagrać z każdym
        for genome_id2, genome2 in genomes[min(i + 1, len(genomes) - 1):]:
            # wartość_if_true IF warunek ELSE wartość_IF_FALSE
            genome2.fitness = 0 if genome2.fitness is None else genome2.fitness
            pong = PongGame(win, width, height)
            force_quit = pong.train_single_ai_pair(genome1, genome2, cfg, True)
            if force_quit:
                quit()


def test_best(cfg):
    # Testowanie najlepszej sieci neuronowej na podstawie wcześniej zapisanego pliku python obj
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
    # tworzenie sieci z najlepszego genomu
    winner_net = neat.nn.FeedForwardNetwork.create(winner, cfg)

    width, height = 700, 500
    win = pygame.display.set_mode((width, height))

    pong = PongGame(win, width, height)
    pong.testing_ai(winner_net)


def neat_runner(cfg):
    # checkpoint z którego chcemy startować (jeśli odkomentujemy trzeba zakomentować neat.Population)
    population = neat.Checkpointer.restore_checkpoint('neat-checkpoint-37')
    # population: Population = neat.Population(cfg)
    population.add_reporter(neat.StdOutReporter(True))  # reporting przez konsole
    population.add_reporter(neat.Checkpointer(1))

    winn = population.run(eval_genomes, 50)
    # eval_genomes jest wywoływana dla każdej generacji populacji, a liczba 50 określa, ile generacji ma być przetwarzanych.
    with open("best.pickle", "wb") as f:
        pickle.dump(winn, f)


def main():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    # https://neat-python.readthedocs.io/en/latest/config_file.html
    # Neat posiada defaultowe wartości na których sieć neuronowa będzie operować
    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
                      config_path)
    neat_runner(cfg)
    test_best(cfg)


if __name__ == '__main__':
    main()
