import kivy
from kivy.app import App
from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.uix.screenmanager import ScreenManager
from kivymd.uix.screen import MDScreen
from kivy.properties import ObjectProperty
from kivy.uix.behaviors import DragBehavior
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.clock import Clock

class LoadingScreen(MDScreen):
    pass


class HomeScreen(MDScreen):
    pass


class AboutScreen(MDScreen):
    pass


class DeveloperScreen(MDScreen):
    pass

class WindowManager(ScreenManager):
    screen_manager = ObjectProperty()


class MyApp(MDApp):
    def build(self):
        global screen_manager
        screen_manager = ScreenManager()
        screen_manager.add_widget(Builder.load_file("loading.kv"))
        screen_manager.add_widget(Builder.load_file("style.kv"))
        return screen_manager

    def on_start(self):
        Clock.schedule_once(self.load_screen,5)

    def load_screen(self,*args):
        screen_manager.current = "main_screen"

if __name__ == "__main__":
    MyApp().run()
