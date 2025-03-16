import { ComponentFixture, TestBed } from '@angular/core/testing';

import { RecentProjectsComponentComponent } from './recent-projects-component.component';

describe('RecentProjectsComponentComponent', () => {
  let component: RecentProjectsComponentComponent;
  let fixture: ComponentFixture<RecentProjectsComponentComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [RecentProjectsComponentComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(RecentProjectsComponentComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
