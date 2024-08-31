import pygame
import random
from pygame import mixer
import torch
import torch.nn as nn
import torch.optim as optim



# Initialize Pygame
pygame.init()

# Set up the game window
window_width = 800
window_height = 600
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Human vs Robot on Pong Game")

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


class Button:
    def __init__(self, x, y, width, height, text, font, color=WHITE, background=BLACK):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font
        self.color = color
        self.background = background

    def draw(self, window):
        pygame.draw.rect(window, self.background, self.rect)
        text_surface = self.font.render(self.text, True, self.color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        window.blit(text_surface, text_rect)

    def is_clicked(self, position):
        return self.rect.collidepoint(position)
class PaddleController(nn.Module):
    def __init__(self):
        super(PaddleController, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # Input layer
        self.fc2 = nn.Linear(64, 1)  # Output layer
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class Ball:
    def __init__(self, x, y, radius, speed_x, speed_y):
        self.x = x
        self.y = y
        self.radius = radius
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.reset_position = (x, y)  # Store the initial position for reset

    def move(self):
        self.x += self.speed_x
        self.y += self.speed_y

    def draw(self, window):
        pygame.draw.circle(window, WHITE, (self.x, self.y), self.radius)

    def check_collision(self, paddle1, paddle2, game):
        # Top and bottom walls
        if self.y <= self.radius or self.y >= window_height - self.radius:
            self.speed_y *= -1

        # Paddle collisions
        if (
            not (not (self.x <= paddle1.x + paddle1.width + self.radius) or not (self.y >= paddle1.y))
            and self.y <= paddle1.y + paddle1.height
        ):
            self.speed_x *= -1
            game.collision_sound.play()

        elif (
                not (not (self.x >= paddle2.x - self.radius) or not (self.y >= paddle2.y))
                and self.y <= paddle2.y + paddle2.height
        ):
            self.speed_x *= -1
            game.collision_sound.play()
        else:
            # Check if ball goes out of the screen
            if self.x <= 0:
                game.score_ai += 1  # AI scores when the ball goes out on the left
                self.reset(game)
            elif self.x >= window_width:
                game.score_player += 1  # Player scores when the ball goes out on the right
                self.reset(game)

    def reset(self, game):
        self.x = window_width // 2
        self.y = random.randint(self.radius, window_height - self.radius)
        self.speed_x = -self.speed_x
class Paddle:
    def __init__(self, x, y, width, height, speed, is_ai = False):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.speed = speed if not is_ai else speed // 2
        self.is_ai = is_ai

    def move_up(self):
        if self.y > 0:
            self.y -= self.speed

    def move_down(self):
        if self.y < window_height - self.height:
            self.y += self.speed

    def draw(self, window):
        paddle_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(window, WHITE, paddle_rect)


class PongGame:
    def __init__(self):
        self.player_paddle = Paddle(window_width - 20, window_height // 2 - 50, 10, 100, 0.6)
        self.ai_paddle = Paddle(10, window_height // 2 - 50, 10, 100, 5, is_ai=True)
        self.ball = Ball(window_width // 2, window_height // 2, 10, 0.8, 0.8)
        self.score_player = 0
        self.score_ai = 0
        self.font = pygame.font.Font(None, 36)
        self.game_active = False
        self.start_button = Button(window_width // 2 - 100, window_height // 3, 200, 50, "Start", self.font)
        self.exit_button = Button(window_width // 2 - 100, window_height // 3 * 2, 200, 50, "Exit", self.font)
        self.collision_sound = mixer.Sound("pong_sound.wav")

        # Initialize the neural network
        self.paddle_controller = PaddleController()
        self.optimizer = optim.Adam(self.paddle_controller.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN and not self.game_active:
                if self.start_button.is_clicked(event.pos):
                    self.game_active = True
                elif self.exit_button.is_clicked(event.pos):
                    return False

        if self.game_active:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                self.player_paddle.move_up()
            if keys[pygame.K_DOWN]:
                self.player_paddle.move_down()

            # The AI paddle
            """if self.ai_paddle.y + self.ai_paddle.height // 2 < self.ball.y:
                self.ai_paddle.move_down()
                
            else:
                self.ai_paddle.move_up()
                print('down')"""

        return True

    def update(self):
        if self.game_active:
            self.ball.move()
            self.ball.check_collision(self.ai_paddle, self.player_paddle, self)

            # Get the current state
            state = torch.tensor([self.ball.y, self.ai_paddle.y], dtype=torch.float32)

            # Get the predicted paddle movement
            predicted_movement = self.paddle_controller(state)

            # Move the AI paddle based on the prediction
            if predicted_movement > 0:
                self.ai_paddle.move_down()
            else:
                self.ai_paddle.move_up()

            # Update the neural network based on the reward
            reward = 0
            if self.ball.speed_x < 0:  # Ball is moving towards the AI paddle
                reward = 1 if self.ball.y >= self.ai_paddle.y and self.ball.y <= self.ai_paddle.y + self.ai_paddle.height else -1

            target = torch.tensor([reward], dtype=torch.float32)
            loss = self.criterion(predicted_movement, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    def draw(self):
        window.fill(BLACK)

        if not self.game_active:
            self.start_button.draw(window)
            self.exit_button.draw(window)
        else:
            # Game logic when active
            self.player_paddle.draw(window)
            self.ai_paddle.draw(window)
            self.ball.draw(window)
            player_score_text = self.font.render(str(self.score_player), True, WHITE)
            ai_score_text = self.font.render(str(self.score_ai), True, WHITE)
            window.blit(player_score_text, (window_width // 4, 10))
            window.blit(ai_score_text, (window_width * 3 // 4, 10))

        pygame.display.update()

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()

        pygame.quit()


# Create a new game instance and start the game loop
game = PongGame()
game.run()