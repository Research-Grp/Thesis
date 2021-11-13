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

class LoadingScreen(MDScreen):
    pass
class HomeScreen(MDScreen):
    pass
class AboutScreen(MDScreen):
    pass
class DeveloperScreen(MDScreen):
    pass

class WindowManager(ScreenManager):
    screen_manager = ObjectProperty(None)


class DragLabel(DragBehavior, Label):
    pass


class MyApp(MDApp):
    def build(self):
        Window.bind(on_dropfile=self._on_file_drop)
        self.root_widget = Builder.load_file("style.kv")
        return self.root_widget

    def _on_file_drop(self, window, file_path):
        print(file_path)
        return

if __name__ == "__main__":
    MyApp().run()
