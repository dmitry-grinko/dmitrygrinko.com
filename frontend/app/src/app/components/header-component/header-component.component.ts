import { Component, OnInit, HostListener } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { ContentService } from '../../services/content.service';

interface MenuItem {
  label: string;
  link: string;
  exact?: boolean;
  isCTA?: boolean;
  submenu?: MenuItem[];
}

@Component({
  selector: 'app-header-component',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './header-component.component.html',
  styleUrl: './header-component.component.scss'
})
export class HeaderComponentComponent implements OnInit {
  isMobileMenuOpen = false;
  headerContent: { title: string; links: MenuItem[] } = { title: '', links: [] };
  activeSubmenu: MenuItem | null = null;

  constructor(private readonly contentService: ContentService) { }

  ngOnInit(): void {
    this.headerContent = this.contentService.getHeader();
  }
  
  toggleMobileMenu() {
    this.isMobileMenuOpen = !this.isMobileMenuOpen;
  }

  onMenuItemHover(link: MenuItem) {
    if (link.submenu) {
      this.activeSubmenu = link;
    } else {
      this.activeSubmenu = null;
    }
  }

  @HostListener('document:click')
  closeSubmenu() {
    this.activeSubmenu = null;
  }
}
