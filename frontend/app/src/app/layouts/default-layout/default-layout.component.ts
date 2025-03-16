import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { HeaderComponentComponent } from '../../components/header-component/header-component.component';
import { FooterComponentComponent } from '../../components/footer-component/footer-component.component';  


@Component({
  selector: 'app-default-layout',
  imports: [
    RouterOutlet,
    HeaderComponentComponent,
    FooterComponentComponent
  ],
  templateUrl: './default-layout.component.html',
  styleUrl: './default-layout.component.scss'
})
export class DefaultLayoutComponent {

}
