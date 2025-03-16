import { Component } from '@angular/core';
import { ContentService } from '../../services/content.service';
import { NgFor } from '@angular/common';

@Component({
  selector: 'app-footer-component',
  standalone: true,
  imports: [NgFor],
  templateUrl: './footer-component.component.html',
  styleUrls: ['./footer-component.component.scss']
})
export class FooterComponentComponent {
  currentYear: number = new Date().getFullYear();
  footerContent;

  constructor(private contentService: ContentService) {
    this.footerContent = this.contentService.getFooter();
  }
}
