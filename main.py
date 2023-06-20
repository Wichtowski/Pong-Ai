import pygame as py
py.init()

WIDTH, HEIGHT = 700, 500
WIN = py.display.set_mode((WIDTH, HEIGHT)) 
py.display.set_caption("Pong")
FPS = 60

def draw(window):
    window.fill((18, 20, 32))
    py.display.update()

def main():
    run = True
    framerate_limiter = py.time.Clock()
    
    while run:
        framerate_limiter.tick(FPS)
        draw(WIN)
        # all event handler
        for event in py.event.get():            
            if event.type == py.QUIT:
                run = False
                break
    
            #(115, 29, 216)


# best practice to work with files
if __name__ == "__main__":   
    main()