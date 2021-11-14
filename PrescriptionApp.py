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
    def _on_file_drop(self, window, file_path):
        print(file_path)
        self.filePath = file_path.decode("utf-8")  # convert byte to string
        self.ids.img.source = self.filePath
        self.ids.img.reload()  # reload image


class AboutScreen(MDScreen):
    pass


class DeveloperScreen(MDScreen):
    pass

class WindowManager(ScreenManager):
    screen_manager = ObjectProperty()


class MyApp(MDApp):
    def build(self):
        screen_manager = ScreenManager()
        screen_manager.add_widget(Builder.load_file("style.kv"))
        screen_manager.add_widget(Builder.load_file("loading.kv"))
        return screen_manager



if __name__ == "__main__":
    MyApp().run()
