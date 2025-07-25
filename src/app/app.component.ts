import { Component, OnInit } from '@angular/core';
import { HeaderComponent } from './components/header/header.component';
import { LayoutComponent } from './components/layout/layout.component';
import { ThemeService } from './services/theme.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [HeaderComponent, LayoutComponent],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {
  title = 'Dmitry Grinko Blog';

  constructor(private themeService: ThemeService) {}

  ngOnInit() {
    // Initialize theme system
    this.themeService.initSystemThemeListener();
  }
}
