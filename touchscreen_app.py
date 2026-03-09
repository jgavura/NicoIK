import pygame
import ctypes
import numpy as np
from tablet_coords_conversion import sim2tab


WORKING_AREA_X = 0.45
WORKING_AREA_Y = 0.20
WORKING_AREA_X_END = 0.30
WORKING_AREA_Y_END = -0.20

TARGETS_FILE = "experiment_grasping/targets_result.txt"
TARGETS_RESULT_FILE = "experiment_grasping/targets_result.txt"


class Touchscreen_app:
    def __init__(self):
        # create a Pygame window
        pygame.init()
        sizes = pygame.display.get_desktop_sizes()
        self.width, self.height = sizes[1]
        self.screen = pygame.display.set_mode((self.width, self.height), flags=pygame.NOFRAME, depth=0, display=1)
        pygame.display.set_caption('NICOs touchscreen')

        # set the window in a way that it never gets focus
        HWND = pygame.display.get_wm_info()['window']
        GWL_EXSTYLE = -20
        styles = ctypes.windll.user32.GetWindowLongA(HWND,GWL_EXSTYLE)
        WS_EX_NOACTIVATE = 0x08000000
        styles |= WS_EX_NOACTIVATE
        ctypes.windll.user32.SetWindowLongA(HWND,GWL_EXSTYLE,styles)

        self.targets = []
        self.target_index = 0

        self.objects = {}
        self.buttons = {}

        # show working area of nn (real - results in pixels)
        # self.objects['working_area'] = Rectangle((162, 100, 1638, 605), (0, 255, 0), width=3)

        # show working area of nn (sim)
        x, y = sim2tab(WORKING_AREA_X, WORKING_AREA_Y)
        x_end, y_end = sim2tab(WORKING_AREA_X_END, WORKING_AREA_Y_END)
        working_area = Rectangle(
            rect=(x, y, x_end-x, y_end-y),
            color_outline=(0, 255, 0),
            outline_width=3
        )
        working_area.add_text("Neural network 2D correction working area", 'corner', (0, 255, 0), 30, 0, 30)
        working_area.add_text(f"Showing target {self.target_index+1}", 'corner', (0, 255, 0), 30, -1425, 30)
        self.objects['working_area'] = working_area

        self.load_targets(TARGETS_FILE)

        target_sim_pos = self.targets[self.target_index][0]
        target = Target(center=sim2tab(target_sim_pos[0], target_sim_pos[1]))
        self.objects['target'] = target

        previous_target_button = Rectangle(
            rect=(self.width * 0.2, self.height * 0.1, self.width * 0.1, self.height * 0.075),
            color_outline=(255, 165, 0),
            color_fill=(255, 200, 120),
            outline_width=3
        )
        previous_target_button.add_text("Previous", 'center', (255, 165, 0), 50)
        self.buttons['previous_target'] = previous_target_button

        next_target_button = Rectangle(
            rect=(self.width * 0.09, self.height * 0.1, self.width * 0.1, self.height * 0.075),
            color_outline=(255, 165, 0),
            color_fill=(255, 200, 120),
            outline_width=3
        )
        next_target_button.add_text("Next", 'center', (255, 165, 0), 50)
        self.buttons['next_target'] = next_target_button

        fail_button = Rectangle(
            rect=(self.width * 0.2, self.height * 0.2, self.width * 0.1, self.height * 0.075),
            color_outline=(255, 165, 0),
            color_fill=(255, 200, 120),
            outline_width=3
        )
        fail_button.add_text("Failure", 'center', (255, 165, 0), 50)
        self.buttons['fail'] = fail_button

        success_button = Rectangle(
            rect=(self.width * 0.09, self.height * 0.2, self.width * 0.1, self.height * 0.075),
            color_outline=(255, 165, 0),
            color_fill=(255, 200, 120),
            outline_width=3
        )
        success_button.add_text("Success", 'center', (255, 165, 0), 50)
        self.buttons['success'] = success_button

        self.update()
    
    def update(self):
        self.screen.fill((0, 0, 0))

        for object in self.objects.values():
            object.draw(self.screen)
        
        for button in self.buttons.values():
            button.draw(self.screen)
        
        self.draw_table()

        pygame.display.flip()
    
    def wait_for_grasp(self):
        if 'grasping' in self.objects:
            self.objects.pop('grasping')
        if 'dropping' in self.objects:
            self.objects.pop('dropping')

        grasp_button = Rectangle(
            rect=(self.width*0.4, self.height*0.1, self.width*0.2, self.height*0.15),
            color_outline=(255, 0, 0),
            color_fill=(128, 0, 0),
            outline_width=3
        )
        grasp_button.add_text("Grasp", 'center', (255, 0, 0), 50)
        self.buttons['grasp'] = grasp_button
        self.update()

        self.wait_for_touch('grasp')
    
    def wait_for_drop(self):
        if 'grasping' in self.objects:
            self.objects.pop('grasping')
        if 'dropping' in self.objects:
            self.objects.pop('dropping')

        drop_button = Rectangle(
            rect=(self.width*0.4, self.height*0.1, self.width*0.2, self.height*0.15),
            color_outline=(255, 0, 0),
            color_fill=(128, 0, 0),
            outline_width=3
        )
        drop_button.add_text("Drop", 'center', (255, 0, 0), 50)
        self.buttons['drop'] = drop_button
        self.update()

        self.wait_for_touch('drop')
    
    def print_touch_pos(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.FINGERDOWN:
                    event_pix_pos = (int(event.x * self.width), int(event.y * self.height))
                    print(f"Finger down event detected at: {event_pix_pos}")
    
    def load_targets(self, filename):
        targets = []

        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split(' ')

                if len(parts) >= 2:
                    x = float(parts[0])
                    y = float(parts[1])
                    grasped = parts[2]
                    targets.append([(x, y), grasped])

        self.targets = targets
    
    def wait_for_touch(self, mode):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.FINGERDOWN:
                    event_pix_pos = (int(event.x * self.width), int(event.y * self.height))
                    print(f"Finger down event detected at: {event_pix_pos}")

                    if mode == 'grasp' and self.buttons['grasp'].hit(event_pix_pos):
                        print("Grasp button pressed")
                        self.buttons.pop('grasp')
                        grasping = Rectangle((self.width*0.5, self.height*0.175, 0, 0), (0, 0, 0))
                        grasping.add_text("Grasping...", 'center', (255, 0, 0), 50)
                        self.objects['grasping'] = grasping
                        self.update()
                        return
                    
                    elif mode == 'drop' and self.buttons['drop'].hit(event_pix_pos):
                        print("Drop button pressed")
                        self.buttons.pop('drop')
                        dropping = Rectangle((self.width*0.5, self.height*0.175, 0, 0), (0, 0, 0))
                        dropping.add_text("Dropping and initializing...", 'center', (255, 0, 0), 50)
                        self.objects['dropping'] = dropping
                        self.update()
                        return

                    elif self.buttons['previous_target'].hit(event_pix_pos):
                        print("Previous target button pressed")
                        self.target_index = (self.target_index - 1) % len(self.targets)
                        self.update_target()
                    
                    elif self.buttons['next_target'].hit(event_pix_pos):
                        print("Next target button pressed")
                        self.target_index = (self.target_index + 1) % len(self.targets)
                        self.update_target()
                    
                    elif self.buttons['fail'].hit(event_pix_pos):
                        print("Failure button pressed")
                        if self.targets[self.target_index][1] != 'False':
                            self.targets[self.target_index][1] = 'False'
                            with open(TARGETS_RESULT_FILE, "w") as f:
                                for (x, y), grasped in self.targets:
                                    f.write(f"{x} {y} {grasped}\n")
                            self.update()
                    
                    elif self.buttons['success'].hit(event_pix_pos):
                        print("Success button pressed")
                        if self.targets[self.target_index][1] != 'True':
                            self.targets[self.target_index][1] = 'True'
                            with open(TARGETS_RESULT_FILE, "w") as f:
                                for (x, y), grasped in self.targets:
                                    f.write(f"{x} {y} {grasped}\n")
                            self.update()
    
    def update_target(self):
        target_sim_pos = self.targets[self.target_index][0]
        target = Target(center=sim2tab(target_sim_pos[0], target_sim_pos[1]))
        self.objects['target'] = target
        self.objects['working_area'].texts.pop(-1)
        self.objects['working_area'].add_text(f"Showing target {self.target_index+1}", 'corner', (0, 255, 0), 30, -1425, 30)
        self.update()
    
    def draw_table(self):
        font = pygame.font.Font(None, 24)
        x, y = sim2tab(WORKING_AREA_X, WORKING_AREA_Y_END)
        x -= 50
        y -= 50
        for i, target in enumerate(self.targets):
            grasped = target[1]
            text = f'{i+1}. '
            if i < 9:
                text = '  ' + text
            if grasped == 'True':
                text += 'Success'
            if grasped == 'False':
                text += 'Failure'
            text_surf = pygame.transform.rotate(font.render(text, True, (255, 165, 0)), 180)
            rect = text_surf.get_rect(bottomright=(x - (i // 10) * 150, y - (i % 10) * 30))
            self.screen.blit(text_surf, rect)
            


class Rectangle:
    def __init__(self, rect, color_outline, color_fill=(0, 0, 0), outline_width=3):
        self.rect = pygame.Rect(rect)
        self.color_outline = color_outline
        self.color_fill = color_fill
        self.outline_width = outline_width
        self.texts = []

    def add_text(self, text, pos_point='center', color=(255, 0, 0), size=24, offset_x=0, offset_y=0):
        font = pygame.font.Font(None, size)
        x, y, width, height = self.rect
        if pos_point == 'center':
            x, y = x + width/2, y + height/2
        elif pos_point == 'corner':
            x, y = x + width, y + height
        self.texts.append((text, (x+offset_x, y+offset_y), pos_point, color, font))

    def draw(self, surface):
        pygame.draw.rect(surface, self.color_fill, self.rect, width=0)
        pygame.draw.rect(surface, self.color_outline, self.rect, self.outline_width)

        for text, pos, pos_point, color, font in self.texts:
            text_surf = pygame.transform.rotate(font.render(text, True, color), 180)
            if pos_point == 'center':
                rect = text_surf.get_rect(center=pos)
            elif pos_point == 'corner':
                rect = text_surf.get_rect(bottomright=pos)
            surface.blit(text_surf, rect)
    
    def hit(self, pos):
        return self.rect.collidepoint(pos)


class Target:
    def __init__(self, center, radius=100, color=(255, 165, 0)):
        self.center = center
        self.radius = radius
        self.color = color
        self.texts = []

    def add_text(self, text, pos_point='center', color=(255, 0, 0), size=24, offset_x=0, offset_y=0):
        font = pygame.font.Font(None, size)
        x, y, width, height = self.rect
        if pos_point == 'center':
            x, y = x + width / 2, y + height / 2
        elif pos_point == 'corner':
            x, y = x + width, y + height
        self.texts.append((text, (x + offset_x, y + offset_y), pos_point, color, font))

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, self.center, self.radius)

        for text, pos, pos_point, color, font in self.texts:
            text_surf = pygame.transform.rotate(font.render(text, True, color), 180)
            if pos_point == 'center':
                rect = text_surf.get_rect(center=pos)
            elif pos_point == 'corner':
                rect = text_surf.get_rect(bottomright=pos)
            surface.blit(text_surf, rect)


if __name__ == "__main__":
    Touchscreen_app().print_touch_pos()
