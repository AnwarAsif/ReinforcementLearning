import numpy as np
from PIL import Image
import cv2

world_size = 10

Gold_Reward = 100
Ghost_Penalty = -300
Pit_Penalty = -300
Move_Penalty = -1

id_player = 1
id_gold = 2
id_ghost = 3
id_pit = 4
id_stench = 5
id_breeze = 6
# Item colors
color_code = {1: (255,174,96),
     2: (0,215,255),
     3: (137, 28, 255),
     4: (153,153,153),
     5: (219, 186, 255),
     6: (234,234,234)}

class WumpusWorld:
    def __init__(self):
        self.x = np.random.randint(0, world_size-1)
        self.y = np.random.randint(1, world_size)

    def move(self, x=False, y=False):

        self.x += x
        self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > world_size -1:
            self.x = world_size - 1

        if self.y < 0:
            self.y = 0
        elif self.y > world_size -1:
            self.y = world_size - 1

    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=0)
        elif choice == 1:
            self.move(x=-1, y=0)
        elif choice == 2:
            self.move(x=0, y=1)
        elif choice == 3:
            self.move(x=0, y=-1)

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)


    def Observation(self):
        pass


class Ghost(WumpusWorld):

    def __init__(self):
        self.x = np.random.randint(0, world_size - 1)
        self.y = np.random.randint(1, world_size)
        if (self.x+1) <= world_size:
            env[self.x + 1][self.y] = color_code[id_stench]
        if (self.x - 1) >= 0:
            env[self.x - 1][self.y] = color_code[id_stench]
        if (self.y + 1) <= world_size:
            env[self.x][self.y + 1] = color_code[id_stench]
        if (self.y - 1) >= 0:
            env[self.x][self.y - 1] = color_code[id_stench]

class Pit(WumpusWorld):
    def __init__(self):
        self.x = np.random.randint(0, world_size - 1)
        self.y = np.random.randint(1, world_size)
        if (self.x + 1) <= world_size:
            env[self.x + 1][self.y] = color_code[id_breeze]
        if (self.x - 1) >= 0:
            env[self.x - 1][self.y] = color_code[id_breeze]
        if (self.y + 1) <= world_size:
            env[self.x][self.y + 1] = color_code[id_breeze]
        if (self.y + 1) <= world_size:
            env[self.x][self.y - 1] = color_code[id_breeze]


env = np.zeros((world_size, world_size, 3), dtype=np.uint8)
player = WumpusWorld()

ghost = Ghost()
pit = Pit()
gold = WumpusWorld()

total_reward = 0
# Set Player's starting Position
player.x = 9
player.y = 0
env[player.x][player.y] = color_code[id_player]
while True:

    current_obs = player.Obs()

    # Random bot
    action = np.random.randint(0, 4)
    player.action(action)

    if player.x == ghost.x and player.y == ghost.y:
        reward = Ghost_Penalty
    elif player.x == pit.x and player.y == pit.y:
        reward = Pit_Penalty
    elif player.x == gold.x and player.y == gold.y:
        reward = Gold_Reward
    else:
        reward = Move_Penalty

    new_obs = (player - gold, player - ghost, player - pit)

    env[player.x][player.y] = color_code[id_player]
    env[ghost.x][ghost.y] = color_code[id_ghost]
    env[gold.x][gold.y] = color_code[id_gold]
    env[pit.x][pit.y] = color_code[id_pit]
    img = Image.fromarray(env, 'RGB')
    img = img.resize((200, 200))
    cv2.imshow('Image', np.array(img))
    if reward == Gold_Reward or reward == Ghost_Penalty or reward == Pit_Penalty:
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break
    else:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    total_reward += reward
    if reward == Gold_Reward or reward == Ghost_Penalty or reward == Pit_Penalty:
        font = cv2.FONT_HERSHEY_SIMPLEX
        break
print('Total Reward:',total_reward)


# Code for Manual playing



